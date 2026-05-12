"""
train.py — NQ ML Bot újratanítási script
=========================================
Használat:
    python train.py

Mit csinál:
    1. Letölti az NQ (QQQ proxy), SPY, VIX historikus napi adatokat yfinance-ről
    2. Felépíti ugyanazokat a feature-öket mint az app.py
    3. XGBoost modellt tanít (következő nap > 0.3% hozam = LONG)
    4. Elmenti: nq_daily_model.pkl
    5. Kiírja a teszt accuracy-t és classification report-ot

Ajánlott futtatás: havonta egyszer, vagy nagy piaci esemény után.
"""

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Konfiguráció ────────────────────────────────────────────────────────────
PERIOD        = "5y"          # Historikus adat (5 év ajánlott)
TARGET_RET    = 0.003         # 0.3% feletti hozam = LONG (1), alatta = SHORT (0)
N_SPLITS      = 5             # TimeSeriesSplit cross-validation
MODEL_FILE    = "nq_daily_model.pkl"

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20",
    "hl_range","rsi","rsi_trend","bb_pos",
    "vol_z","vwap_dev","qqq_spy","vix","vix_ratio","dow"
]

# ── Adatok letöltése ─────────────────────────────────────────────────────────
print("📥 Adatok letöltése yfinance-ről...")
qqq = yf.download("QQQ", period=PERIOD, interval="1d", auto_adjust=True, progress=False)
spy = yf.download("SPY", period=PERIOD, interval="1d", auto_adjust=True, progress=False)
vix = yf.download("^VIX", period=PERIOD, interval="1d", auto_adjust=True, progress=False)

for df in [qqq, spy, vix]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

print(f"✅ QQQ sorok: {len(qqq)} | SPY sorok: {len(spy)} | VIX sorok: {len(vix)}")

# ── Feature engineering ──────────────────────────────────────────────────────
print("⚙️  Feature-ök számítása...")
df = qqq.copy()

for n in [1, 2, 3, 5, 10, 20]:
    df[f"ret_{n}"] = df["Close"].pct_change(n)

df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"]       = 100 - 100 / (1 + gain / (loss + 1e-9))
df["rsi_trend"] = df["rsi"] - df["rsi"].shift(3)

bb_mid      = df["Close"].rolling(20).mean()
bb_std      = df["Close"].rolling(20).std()
df["bb_pos"] = (df["Close"] - bb_mid) / (2 * bb_std + 1e-9)

vol_ma      = df["Volume"].rolling(20).mean()
vol_std     = df["Volume"].rolling(20).std()
df["vol_z"] = (df["Volume"] - vol_ma) / (vol_std + 1e-9)

vwap_n       = (df["Close"] * df["Volume"]).rolling(10).sum()
vwap_d       = df["Volume"].rolling(10).sum()
vwap         = vwap_n / (vwap_d + 1e-9)
df["vwap_dev"] = (df["Close"] - vwap) / (vwap + 1e-9)

df["qqq_spy"]   = df["Close"].pct_change(3) - spy["Close"].reindex(df.index).pct_change(3)
df["vix"]       = vix["Close"].reindex(df.index)
df["vix_ratio"] = df["vix"] / (df["vix"].rolling(10).mean() + 1e-9)
df["dow"]       = df.index.dayofweek

# ── Target: következő nap hozama ─────────────────────────────────────────────
df["target"] = (df["Close"].shift(-1).pct_change(1).shift(-1) > TARGET_RET).astype(int)
# Pontosabban: holnap Close > mai Close * (1 + TARGET_RET)
df["target"] = (df["Close"].shift(-1) > df["Close"] * (1 + TARGET_RET)).astype(int)

df.dropna(inplace=True)

X = df[FEATURES].values
y = df["target"].values

print(f"✅ Tanítóadatok: {len(X)} sor | LONG arány: {y.mean():.1%}")

# ── TimeSeriesSplit cross-validation ────────────────────────────────────────
print(f"\n📊 Cross-validation ({N_SPLITS} fold, time-series)...")
tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
cv_accs = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    cv_accs.append(acc)
    print(f"  Fold {fold+1}: accuracy = {acc:.3f}")

print(f"\n✅ Átlag CV accuracy: {np.mean(cv_accs):.3f} ± {np.std(cv_accs):.3f}")

# ── Végső modell — teljes adaton ─────────────────────────────────────────────
print("\n🏋️  Végső modell tanítása teljes adaton...")
split   = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

final_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
print(f"\n📈 Test accuracy (utolsó 15%): {accuracy_score(y_test, y_pred):.3f}")
print("\n" + classification_report(y_test, y_pred, target_names=["SHORT","LONG"]))

# ── Feature importance ───────────────────────────────────────────────────────
print("🔍 Top feature-ök (importance):")
importances = pd.Series(final_model.feature_importances_, index=FEATURES)
for feat, imp in importances.sort_values(ascending=False).head(8).items():
    print(f"  {feat:15s}: {imp:.4f}")

# ── Mentés ───────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODEL_FILE)
print(f"\n💾 Modell elmentve: {MODEL_FILE}")
print("✅ Kész! Töltsd fel a friss nq_daily_model.pkl-t a GitHub repo-ba.")
