from flask import Flask, request, jsonify
import joblib, numpy as np, os
import pandas as pd
from datetime import datetime, timedelta

app   = Flask(__name__)
model = joblib.load("nq_daily_model.pkl")

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20",
    "hl_range","rsi","rsi_trend","bb_pos",
    "vol_z","vwap_dev","qqq_spy","vix","vix_ratio","dow"
]

def next_trading_day(dt):
    """Returns the next business day after dt (skips weekends)."""
    next_day = dt + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day

def strip_tz(idx):
    """Remove timezone from DatetimeIndex regardless of tz type."""
    try:
        return idx.tz_localize(None)
    except TypeError:
        return idx.tz_convert(None).tz_localize(None)

def drop_incomplete_candle(df):
    """
    yfinance sometimes includes today's still-open candle.
    Drop it by keeping only rows where the date < today (UTC date).
    """
    idx = df.index
    if idx.tz is not None:
        idx = strip_tz(idx)
    today = pd.Timestamp.utcnow().normalize()
    # Keep only strictly past trading days
    mask = idx.normalize() < today
    return df.loc[mask]

def get_latest_features():
    import yfinance as yf

    qqq = yf.download("QQQ", period="90d", interval="1d",
                      auto_adjust=True, progress=False)
    spy = yf.download("SPY", period="90d", interval="1d",
                      auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period="90d", interval="1d",
                      auto_adjust=True, progress=False)

    for frame in [qqq, spy, vix]:
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

    # Drop incomplete (today's intraday) candle from all three
    qqq = drop_incomplete_candle(qqq)
    spy = drop_incomplete_candle(spy)
    vix = drop_incomplete_candle(vix)

    if len(qqq) < 25:
        raise ValueError(f"Not enough QQQ rows after filtering: {len(qqq)}")

    df = qqq.copy()
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}"] = df["Close"].pct_change(n)

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]       = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["rsi_trend"] = df["rsi"] - df["rsi"].shift(3)

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_pos"] = (df["Close"] - bb_mid) / (2 * bb_std + 1e-9)

    vol_ma  = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["vol_z"] = (df["Volume"] - vol_ma) / (vol_std + 1e-9)

    vwap_n = (df["Close"] * df["Volume"]).rolling(10).sum()
    vwap_d = df["Volume"].rolling(10).sum()
    vwap   = vwap_n / (vwap_d + 1e-9)
    df["vwap_dev"] = (df["Close"] - vwap) / (vwap + 1e-9)

    df["qqq_spy"] = df["Close"].pct_change(3) - spy["Close"].reindex(df.index).pct_change(3)

    df["vix"]       = vix["Close"].reindex(df.index)
    df["vix_ratio"] = df["vix"] / (df["vix"].rolling(10).mean() + 1e-9)
    df["dow"]       = df.index.dayofweek

    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("DataFrame empty after dropna. Check yfinance data.")

    row             = df[FEATURES].iloc[-1]
    last_close_date = df.index[-1]
    if hasattr(last_close_date, 'to_pydatetime'):
        last_close_date = last_close_date.to_pydatetime()
    prediction_date = next_trading_day(last_close_date).strftime("%Y-%m-%d")
    return row.to_dict(), prediction_date


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "NQ ML Bot online ✅"})


@app.route("/debug", methods=["GET"])
def debug():
    """Shows raw yfinance row counts before and after filtering."""
    try:
        import yfinance as yf
        qqq_raw = yf.download("QQQ", period="90d", interval="1d",
                              auto_adjust=True, progress=False)
        if isinstance(qqq_raw.columns, pd.MultiIndex):
            qqq_raw.columns = qqq_raw.columns.get_level_values(0)
        qqq_filtered = drop_incomplete_candle(qqq_raw)
        return jsonify({
            "qqq_raw_rows":      len(qqq_raw),
            "qqq_filtered_rows": len(qqq_filtered),
            "qqq_last_raw":      str(qqq_raw.index[-1]),
            "qqq_last_filtered": str(qqq_filtered.index[-1]) if len(qqq_filtered) else "EMPTY",
            "utc_now":           str(pd.Timestamp.utcnow()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/signal", methods=["GET"])
def signal():
    try:
        features, date = get_latest_features()
        X      = np.array([[features[f] for f in FEATURES]])
        proba  = float(model.predict_proba(X)[0][1])
        sig    = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        return jsonify({
            "date":       date,
            "signal":     sig,
            "confidence": round(proba, 4),
            "features":   {k: round(v, 4) for k, v in features.items()}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        features, date = get_latest_features()
        X      = np.array([[features[f] for f in FEATURES]])
        proba  = float(model.predict_proba(X)[0][1])
        sig    = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        print(f"📡 {date} | {sig} | conf: {proba:.2%}")
        return jsonify({"signal": sig, "confidence": round(proba, 4), "date": date})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
