from flask import Flask, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
model    = joblib.load("nq_daily_model.pkl")
FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20",
    "hl_range","rsi","rsi_trend","bb_pos",
    "vol_z","vwap_dev","qqq_spy",
    "vix","vix_ratio","dow"
]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "NQ ML Bot online ✅", "model": "daily_v1", "features": len(FEATURES)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d        = request.json
        features = np.array([[d[f] for f in FEATURES]])
        proba    = float(model.predict_proba(features)[0][1])
        signal   = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        return jsonify({"signal": signal, "confidence": round(proba, 4), "long_prob": round(proba, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        d        = request.json
        features = np.array([[d[f] for f in FEATURES]])
        proba    = float(model.predict_proba(features)[0][1])
        signal   = "LONG" if proba > 0.52 else ("SHORT" if proba < 0.48 else "NEUTRAL")
        print(f"📡 {signal} | conf: {proba:.2%} | dow: {d.get('dow','?')} | vix: {d.get('vix','?')}")
        return jsonify({"signal": signal, "confidence": round(proba, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
