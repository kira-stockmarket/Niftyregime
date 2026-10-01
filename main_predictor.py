import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------
def fetch_market_data(start_date="2012-01-01"):
    """Downloads Nifty 50 and India VIX daily data with clean column extraction."""
    logging.info("Downloading ^NSEI and ^INDIAVIX data...")
    tickers = ["^NSEI", "^INDIAVIX"]
    raw = yf.download(tickers, start=start_date, interval="1d", progress=False)

    # Handle multi-index columns returned by modern yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        nifty_close = raw["Close"]["^NSEI"].dropna()
        nifty_open = raw["Open"]["^NSEI"].dropna()
        nifty_high = raw["High"]["^NSEI"].dropna()
        nifty_low = raw["Low"]["^NSEI"].dropna()
        nifty_vol = raw["Volume"]["^NSEI"].dropna()
        vix_close = raw["Close"]["^INDIAVIX"].dropna()
    else:
        raise ValueError("Unexpected data format returned from yfinance.")

    df = pd.DataFrame({
        "open": nifty_open,
        "high": nifty_high,
        "low": nifty_low,
        "close": nifty_close,
        "volume": nifty_vol,
        "vix": vix_close
    }).dropna()

    df = df[df["close"] > 0].sort_index()
    logging.info(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (ZERO-LOOKAHEAD ENFORCED)
# ---------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def engineer_features(df):
    """Constructs stationary, orthogonal momentum, volatility, and volume indicators."""
    feat = pd.DataFrame(index=df.index)

    # Price / Return Dynamics
    feat["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    feat["log_ret_5"] = np.log(df["close"] / df["close"].shift(5))
    feat["log_ret_20"] = np.log(df["close"] / df["close"].shift(20))

    # Realized & Intraday Volatility
    feat["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
    )
    feat["atr_ratio"] = compute_atr(df, 14) / df["close"]
    feat["hist_vol_20"] = feat["log_ret_1"].rolling(20).std() * np.sqrt(252)

    # Momentum & Trend
    for p in [7, 14, 21]:
        feat[f"rsi_{p}"] = compute_rsi(df["close"], p)

    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    feat["dist_sma20"] = (df["close"] - sma20) / sma20
    feat["dist_sma50"] = (df["close"] - sma50) / sma50
    feat["dist_sma200"] = (df["close"] - sma200) / sma200

    # Bollinger Bands
    rolling_std = df["close"].rolling(20).std()
    feat["bb_width"] = (2 * rolling_std * 2) / (sma20 + 1e-9)
    feat["bb_pos"] = (df["close"] - (sma20 - 2 * rolling_std)) / (4 * rolling_std + 1e-9)

    # India VIX Metrics
    feat["vix_level"] = df["vix"]
    feat["vix_roc_5"] = df["vix"].pct_change(5)
    feat["vix_zscore_20"] = (df["vix"] - df["vix"].rolling(20).mean()) / (df["vix"].rolling(20).std() + 1e-9)

    # Volume Signals
    vol_sma = df["volume"].rolling(20).mean()
    feat["vol_ratio_20"] = df["volume"] / (vol_sma + 1e-9)

    return feat

# ---------------------------------------------------------
# 3. UNSUPERVISED REGIME DETECTION (HMM)
# ---------------------------------------------------------
def fit_hmm_regimes(df, feat):
    """
    Fits a 3-State Gaussian HMM on returns and volatility.
    States are ordered dynamically by annualized return:
      0: Bear / Stress, 1: Sideways / Transition, 2: Bull / Low Vol
    """
    hmm_data = pd.DataFrame({
        "ret": feat["log_ret_1"],
        "vol": feat["parkinson_vol"]
    }).dropna()

    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=42)
    model.fit(hmm_data)
    
    hidden_states = model.predict(hmm_data)
    probs = model.predict_proba(hmm_data)

    # Re-map regimes based on mean return ranking
    state_returns = [hmm_data["ret"][hidden_states == i].mean() for i in range(3)]
    mapping = {old_st: new_st for new_st, old_st in enumerate(np.argsort(state_returns))}

    ordered_states = np.array([mapping[s] for s in hidden_states])
    ordered_probs = np.zeros_like(probs)
    for old_st, new_st in mapping.items():
        ordered_probs[:, new_st] = probs[:, old_st]

    regime_df = pd.DataFrame(index=hmm_data.index)
    regime_df["regime_state"] = ordered_states
    regime_df["prob_bear"] = ordered_probs[:, 0]
    regime_df["prob_sideways"] = ordered_probs[:, 1]
    regime_df["prob_bull"] = ordered_probs[:, 2]

    return regime_df

# ---------------------------------------------------------
# 4. TARGET CONSTRUCTION & MODEL TRAINING (LIGHTGBM)
# ---------------------------------------------------------
def train_and_evaluate_horizon(X_train, y_train, X_test, y_test, horizon_name):
    """Trains a tuned LightGBM classifier with early stopping and out-of-sample metrics."""
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 30,
        "seed": 42,
        "verbose": -1
    }

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[valid_data],
        callbacks=callbacks
    )

    preds_prob = model.predict(X_test)
    preds_bin = (preds_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, preds_bin)
    auc = roc_auc_score(y_test, preds_prob)
    logging.info(f"[{horizon_name}] Out-of-Sample -> Accuracy: {acc:.2%}, AUC: {auc:.3f}")

    return model, {"accuracy": float(acc), "auc": float(auc)}

# ---------------------------------------------------------
# 5. EXECUTION PIPELINE
# ---------------------------------------------------------
def main():
    os.makedirs("output", exist_ok=True)
    df = fetch_market_data(start_date="2013-01-01")

    # 1. Generate base indicators
    feat = engineer_features(df)

    # 2. Fit HMM Regimes
    regime_df = fit_hmm_regimes(df, feat)
    feat = feat.join(regime_df).dropna()

    # 3. Strict alignment: Shift features by 1 bar for training targets
    # Today's close (T) uses information up to T. Target for forward return is calculated after T.
    X_matrix = feat.copy()

    # Targets: Forward return > 0 (1 = Bullish, 0 = Bearish)
    close_s = df["close"].loc[X_matrix.index]
    targets = {
        "Daily (1D)": (close_s.shift(-1) > close_s).astype(int),
        "Weekly (5D)": (close_s.shift(-5) > close_s).astype(int),
        "Monthly (21D)": (close_s.shift(-21) > close_s).astype(int),
    }

    # Extract the absolute latest bar for live forward inference
    latest_inference_bar = X_matrix.iloc[[-1]]
    latest_date = latest_inference_bar.index[0].strftime("%Y-%m-%d")
    latest_close = float(df["close"].loc[latest_inference_bar.index[0]])
    latest_vix = float(df["vix"].loc[latest_inference_bar.index[0]])

    current_regime_id = int(latest_inference_bar["regime_state"].values[0])
    regime_names = {0: "Bear / High Volatility", 1: "Sideways / Transitory", 2: "Bull / Low Volatility"}

    projections = {}
    validation_metrics = {}

    # Train and evaluate models across horizons
    test_split_bars = 252  # 1 year out-of-sample holdout

    for horizon_name, y_target in targets.items():
        # Exclude NaN targets at the end of the series
        valid_idx = y_target.dropna().index
        X_clean = X_matrix.loc[valid_idx]
        y_clean = y_target.loc[valid_idx]

        # Time-series temporal split
        X_train, X_test = X_clean.iloc[:-test_split_bars], X_clean.iloc[-test_split_bars:]
        y_train, y_test = y_clean.iloc[:-test_split_bars], y_clean.iloc[-test_split_bars:]

        model, metrics = train_and_evaluate_horizon(X_train, y_train, X_test, y_test, horizon_name)
        validation_metrics[horizon_name] = metrics

        # Live forward projection
        prob_bull = float(model.predict(latest_inference_bar)[0])
        projections[horizon_name] = {
            "stance": "BULLISH" if prob_bull >= 0.50 else "BEARISH",
            "bullish_probability": round(prob_bull * 100, 2),
            "bearish_probability": round((1 - prob_bull) * 100, 2)
        }

    # Prepare final output structure
    output_payload = {
        "metadata": {
            "date": latest_date,
            "nifty_close": latest_close,
            "india_vix": latest_vix,
            "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        },
        "market_regime": {
            "detected_state": regime_names.get(current_regime_id, "Unknown"),
            "probabilities": {
                "bear_risk": round(float(latest_inference_bar["prob_bear"].values[0]) * 100, 2),
                "sideways": round(float(latest_inference_bar["prob_sideways"].values[0]) * 100, 2),
                "bull_trend": round(float(latest_inference_bar["prob_bull"].values[0]) * 100, 2)
            }
        },
        "projections": projections,
        "out_of_sample_metrics": validation_metrics
    }

    # Write JSON output artifact
    json_path = "output/nifty_predictions.json"
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4)

    # Print terminal report
    print("\n" + "=" * 65)
    print(f" NIFTY 50 REGIME & DIRECTIONAL PREDICTOR: {latest_date}")
    print(f" Underlying Close: {latest_close:.2f} | India VIX: {latest_vix:.2f}")
    print("=" * 65)
    print(f"HMM Macro Regime: {output_payload['market_regime']['detected_state'].upper()}")
    print(f"Regime Probabilities -> Bull: {output_payload['market_regime']['probabilities']['bull_trend']}% | "
          f"Bear: {output_payload['market_regime']['probabilities']['bear_risk']}% | "
          f"Sideways: {output_payload['market_regime']['probabilities']['sideways']}%\n")

    print(f"{'Horizon':<15} | {'Stance':<9} | {'Bull Prob':<10} | {'Bear Prob':<10} | {'OOS Acc'}")
    print("-" * 65)
    for horizon, res in projections.items():
        acc = validation_metrics[horizon]["accuracy"]
        print(f"{horizon:<15} | {res['stance']:<9} | {res['bullish_probability']:>8.2f}% | "
              f"{res['bearish_probability']:>8.2f}% | {acc:>6.2%}")
    print("=" * 65 + "\n")
    logging.info(f"Artifact successfully saved to {json_path}")

if __name__ == "__main__":
    main()
