from flask import Flask, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
model = joblib.load("nq_daily_model.pkl")

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20",
    "hl_range","rsi","rsi_trend","bb_pos",
    "vol_z","vwap_dev","qqq_spy",
    "vix","vix_ratio","dow"
]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "NQ ML Bot online ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.json
        features = np.array([[d[f] for f in FEATURES]])
        proba = float(model.predict_proba(features)[0][1])
        signal = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        return jsonify({
            "signal":     signal,
            "confidence": round(proba, 4),
            "threshold":  0.52
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/webhook", methods=["POST"])
def webhook():
    # TradingView Pine Script alert fogadása
    d = request.json
    try:
        features = np.array([[d[f] for f in FEATURES]])
        proba = float(model.predict_proba(features)[0][1])
        signal = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        print(f"📡 Alert érkezett | Signal: {signal} | Conf: {proba:.2%}")
        return jsonify({"signal": signal, "confidence": round(proba, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)