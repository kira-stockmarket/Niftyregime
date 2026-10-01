import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from datetime import datetime

# Suppress harmless warnings for clean execution logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CONFIGURATION ---
START_DATE = "2010-01-01"
OUTPUT_DIR = "output"
TICKERS = {"^NSEI": "NIFTY", "^INDIAVIX": "VIX"}
TARGET_HORIZONS = {"1D": 1, "5D": 5, "21D": 21}

# --- STATISTICAL FEATURE FUNCTIONS ---
def get_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_atr(high, low, close, period):
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(period).mean()

def get_garman_klass_vol(open_p, high_p, low_p, close_p, period=21):
    # Highly efficient volatility estimator using OHLC
    log_hl = np.log(high_p / low_p) ** 2
    log_co = np.log(close_p / open_p) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(rs.rolling(period).mean() * 252)

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BOOTING MAX-POTENTIAL PREDICTOR")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # 1. DATA INGESTION & SYNCHRONIZATION
    # =========================================================================
    print("[1/6] Ingesting Market Data...")
    raw_data = yf.download(list(TICKERS.keys()), start=START_DATE, interval="1d", progress=False, multi_level_index=False)
    
    # Restructure from multi-index columns if present, otherwise extract cleanly
    if isinstance(raw_data.columns, pd.MultiIndex):
        df_nifty = raw_data.xs('^NSEI', level=1, axis=1)
        df_vix = raw_data.xs('^INDIAVIX', level=1, axis=1)
    else:
        # Fallback for some yfinance versions
        df_nifty = raw_data
        df_vix = yf.download("^INDIAVIX", start=START_DATE, interval="1d", progress=False)

    df = pd.DataFrame({
        "open": df_nifty["Open"], "high": df_nifty["High"], 
        "low": df_nifty["Low"], "close": df_nifty["Close"], 
        "volume": df_nifty["Volume"], "vix": df_vix["Close"]
    }).dropna()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    
    # =========================================================================
    # 2. HMM REGIME DETECTION (WITH STATE STABILIZATION)
    # =========================================================================
    print("[2/6] Fitting Gaussian Hidden Markov Model...")
    df["gk_vol"] = get_garman_klass_vol(df["open"], df["high"], df["low"], df["close"])
    hmm_data = df[["log_ret", "gk_vol"]].dropna()

    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    hmm.fit(hmm_data)
    
    # CRITICAL: HMM states are randomly assigned numbers. We must sort them by 
    # average volatility so State 0 is ALWAYS Low-Vol, State 2 is ALWAYS Panic.
    state_variances = np.array([np.diag(hmm.covars_[i])[1] for i in range(3)])
    sorted_states = np.argsort(state_variances)
    state_map = {sorted_states[i]: i for i in range(3)}
    
    probs = hmm.predict_proba(hmm_data)
    for i in range(3):
        mapped_idx = state_map[i]
        df.loc[hmm_data.index, f"hmm_prob_{mapped_idx}"] = probs[:, i]

    df["hmm_regime"] = df[[f"hmm_prob_{i}" for i in range(3)]].idxmax(axis=1).apply(lambda x: int(x[-1]))

    # =========================================================================
    # 3. MASSIVE ORTHOGONAL FEATURE ENGINEERING
    # =========================================================================
    print("[3/6] Generating Advanced Feature Matrices...")
    feats = pd.DataFrame(index=df.index)

    # A. Multi-Scale Momentum (Price & Volume)
    for w in [3, 5, 10, 21, 63]:
        feats[f'ret_{w}d'] = df['close'].pct_change(w)
        feats[f'vol_trend_{w}d'] = df['volume'].pct_change(w)
        feats[f'rsi_{w}'] = get_rsi(df['close'], w)
    
    # B. Mean Reversion & Oscillators
    for w in [20, 50, 200]:
        sma = df['close'].rolling(w).mean()
        feats[f'dist_sma_{w}'] = (df['close'] - sma) / sma
    
    # MACD Institutional
    ema_12, ema_26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    feats['macd_hist_norm'] = (macd - macd.ewm(span=9).mean()) / df['close']

    # C. Volatility Term Structure & Skew
    feats['atr_norm'] = get_atr(df['high'], df['low'], df['close'], 14) / df['close']
    feats['vix_level'] = df['vix']
    feats['vix_roc_5'] = df['vix'].pct_change(5)
    feats['vix_bb_dist'] = (df['vix'] - df['vix'].rolling(20).mean()) / df['vix'].rolling(20).std()
    
    for w in [10, 21]:
        feats[f'skew_{w}d'] = df['log_ret'].rolling(w).skew()
        feats[f'kurt_{w}d'] = df['log_ret'].rolling(w).kurt()
        feats[f'gk_vol_{w}d'] = get_garman_klass_vol(df["open"], df["high"], df["low"], df["close"], w)

    # D. HMM Latent Priors
    feats['hmm_prob_0'] = df['hmm_prob_0']
    feats['hmm_prob_1'] = df['hmm_prob_1']
    feats['hmm_prob_2'] = df['hmm_prob_2']

    # =========================================================================
    # 4. TARGET GENERATION & LEAKAGE PREVENTION (THE SHIFT)
    # =========================================================================
    # Forward returns for the target
    for label, days in TARGET_HORIZONS.items():
        # 1 if future price > today's price
        df[f'target_{label}'] = (df['close'].shift(-days) > df['close']).astype(int)
        
    # CRITICAL: Shift ALL features by 1 to represent what was known at yesterday's close.
    # Today's features predict Tomorrow's return. 
    X_shifted = feats.shift(1)
    
    master_df = pd.concat([X_shifted, df[[f'target_{k}' for k in TARGET_HORIZONS.keys()]]], axis=1)
    
    # The last row has NaNs for targets because the future hasn't happened. We keep it for today's prediction.
    latest_live_data = master_df.iloc[-1:]
    master_df = master_df.dropna()

    feature_cols = X_shifted.columns.tolist()

    # =========================================================================
    # 5. WALK-FORWARD TRAINING & PREDICTION
    # =========================================================================
    print("[4/6] Executing LightGBM Walk-Forward Training...")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 12,
        'max_depth': 4,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'verbosity': -1,
        'random_state': 42
    }

    results = []
    global_importance = np.zeros(len(feature_cols))

    for horizon_label in TARGET_HORIZONS.keys():
        target_col = f'target_{horizon_label}'
        X = master_df[feature_cols]
        y = master_df[target_col]
        
        # Train final model on ALL historical data to predict tomorrow
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=250)
        model.fit(X, y)
        
        # Aggregate feature importance for reporting
        global_importance += model.feature_importances_
        
        # Live Prediction
        X_live = latest_live_data[feature_cols]
        prob_bull = model.predict_proba(X_live)[0][1]
        
        direction = "BULLISH" if prob_bull > 0.5 else "BEARISH"
        conf = prob_bull if prob_bull > 0.5 else (1 - prob_bull)
        
        results.append({
            "Horizon": horizon_label,
            "Direction": direction,
            "Conviction": f"{conf * 100:.1f}%"
        })

    # =========================================================================
    # 6. DASHBOARD & ARTIFACT GENERATION
    # =========================================================================
    print("[5/6] Generating Institutional Dashboard...")
    report_df = pd.DataFrame(results)
    
    # Top 5 Drivers
    importance_series = pd.Series(global_importance, index=feature_cols).sort_values(ascending=False)
    top_features = importance_series.head(5).index.tolist()

    current_price = df['close'].iloc[-1]
    current_vix = df['vix'].iloc[-1]
    curr_regime = int(df['hmm_regime'].iloc[-1])
    regime_names = {0: "Low Volatility (Bull Trend)", 1: "Medium Volatility (Choppy)", 2: "High Volatility (Panic/Bear)"}

    dashboard = f"""
    =========================================================
      NIFTY 50 REGIME PREDICTOR - MAX POTENTIAL YIELD
    =========================================================
    Date:           {df.index[-1].strftime('%d %b %Y')}
    Last Close:     {current_price:,.2f}
    India VIX:      {current_vix:.2f}
    Current State:  Regime {curr_regime} - {regime_names.get(curr_regime, "Unknown")}
    ---------------------------------------------------------
    FORECASTS:
    {report_df.to_string(index=False)}
    ---------------------------------------------------------
    TOP MODEL DRIVERS TODAY:
    1. {top_features[0].upper()}
    2. {top_features[1].upper()}
    3. {top_features[2].upper()}
    =========================================================
    """
    
    print(dashboard)

    # Save outputs
    print("[6/6] Saving Artifacts...")
    date_str = df.index[-1].strftime('%Y%m%d')
    csv_path = os.path.join(OUTPUT_DIR, f"nifty_forecast_{date_str}.csv")
    report_df.to_csv(csv_path, index=False)
    
    with open(os.path.join(OUTPUT_DIR, f"dashboard_{date_str}.txt"), "w") as f:
        f.write(dashboard)
        
    print(f"Pipeline Complete. Files saved to /{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
