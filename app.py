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

def get_latest_features():
    """Alpaca helyett yfinance — nem kell API kulcs a szerveren"""
    import yfinance as yf
    qqq = yf.download("QQQ", period="60d", interval="1d",
                      auto_adjust=True, progress=False)
    spy = yf.download("SPY", period="60d", interval="1d",
                      auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period="60d", interval="1d",
                      auto_adjust=True, progress=False)

    for df in [qqq, spy, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = qqq.copy()
    for n in [1,2,3,5,10,20]:
        df[f"ret_{n}"] = df["Close"].pct_change(n)

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["rsi_trend"]= df["rsi"] - df["rsi"].shift(3)

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_pos"] = (df["Close"] - bb_mid) / (2 * bb_std + 1e-9)

    vol_ma = df["Volume"].rolling(20).mean()
    vol_std= df["Volume"].rolling(20).std()
    df["vol_z"] = (df["Volume"] - vol_ma) / (vol_std + 1e-9)

    vwap_n = (df["Close"] * df["Volume"]).rolling(10).sum()
    vwap_d = df["Volume"].rolling(10).sum()
    vwap   = vwap_n / (vwap_d + 1e-9)
    df["vwap_dev"] = (df["Close"] - vwap) / (vwap + 1e-9)

    df["qqq_spy"] = df["Close"].pct_change(3) - spy["Close"].reindex(df.index).pct_change(3)

    df["vix"]      = vix["Close"].reindex(df.index)
    df["vix_ratio"]= df["vix"] / (df["vix"].rolling(10).mean() + 1e-9)
    df["dow"]      = df.index.dayofweek

    df.dropna(inplace=True)
    row = df[FEATURES].iloc[-1]
    return row.to_dict(), df.index[-1].strftime("%Y-%m-%d")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "NQ ML Bot online ✅"})

@app.route("/signal", methods=["GET"])
def signal():
    """Legutóbbi napi predikció — böngészőből is hívható"""
    try:
        features, date = get_latest_features()
        X      = np.array([[features[f] for f in FEATURES]])
        proba  = float(model.predict_proba(X)[0][1])
        signal = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        return jsonify({
            "date":       date,
            "signal":     signal,
            "confidence": round(proba, 4),
            "features":   {k: round(v, 4) for k, v in features.items()}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """TradingView trigger — maga számolja ki a feature-öket"""
    try:
        features, date = get_latest_features()
        X      = np.array([[features[f] for f in FEATURES]])
        proba  = float(model.predict_proba(X)[0][1])
        signal = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        print(f"📡 {date} | {signal} | conf: {proba:.2%}")
        return jsonify({"signal": signal, "confidence": round(proba, 4), "date": date})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
