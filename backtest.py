import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import lightgbm as lgb
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CONFIGURATION ---
START_DATE = "2010-01-01"
TICKERS = {"^NSEI": "NIFTY", "^INDIAVIX": "VIX"}
INITIAL_TRAIN_DAYS = 1260  # Train on 5 years initially
STEP_DAYS = 126            # Retrain every 6 months

def get_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def get_atr(high, low, close, period):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(period).mean()

def get_garman_klass_vol(open_p, high_p, low_p, close_p, period=21):
    log_hl = np.log(high_p / low_p) ** 2
    log_co = np.log(close_p / open_p) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(rs.rolling(period).mean() * 252)

def calculate_metrics(returns_series, name="Strategy"):
    ann_ret = np.exp(returns_series.mean() * 252) - 1
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    cum_ret = np.exp(returns_series.cumsum())
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    total_ret = cum_ret.iloc[-1] - 1
    
    return {
        "Name": name,
        "Total Return": f"{total_ret:.2%}",
        "Annual Return": f"{ann_ret:.2%}",
        "Annual Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}"
    }

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading Data...")
    raw_data = yf.download(list(TICKERS.keys()), start=START_DATE, interval="1d", progress=False, multi_level_index=False)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        df = pd.DataFrame({"open": raw_data.xs('^NSEI', level=1, axis=1)["Open"], 
                           "high": raw_data.xs('^NSEI', level=1, axis=1)["High"], 
                           "low": raw_data.xs('^NSEI', level=1, axis=1)["Low"], 
                           "close": raw_data.xs('^NSEI', level=1, axis=1)["Close"], 
                           "vix": raw_data.xs('^INDIAVIX', level=1, axis=1)["Close"]}).dropna()
    else:
        df = pd.DataFrame({"open": raw_data["Open"], "high": raw_data["High"], 
                           "low": raw_data["Low"], "close": raw_data["Close"], 
                           "vix": yf.download("^INDIAVIX", start=START_DATE, interval="1d", progress=False)["Close"]}).dropna()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["gk_vol"] = get_garman_klass_vol(df["open"], df["high"], df["low"], df["close"])
    df = df.dropna(subset=["log_ret", "gk_vol"]).copy()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting Global HMM (State 2 = Panic)...")
    hmm_data = df[["log_ret", "gk_vol"]]
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    hmm.fit(hmm_data)
    
    # Sort states: 0 = Low Vol Bull, 1 = Mid Vol Choppy, 2 = High Vol Panic
    state_variances = np.array([np.diag(hmm.covars_[i])[1] for i in range(3)])
    sorted_states = np.argsort(state_variances)
    state_map = {sorted_states[i]: i for i in range(3)}
    
    probs = hmm.predict_proba(hmm_data)
    for i in range(3): 
        mapped_idx = state_map[i]
        df[f"hmm_prob_{mapped_idx}"] = probs[:, i]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Engineering Orthogonal Features...")
    feats = pd.DataFrame(index=df.index)
    for w in [3, 5, 10, 21, 63]:
        feats[f'ret_{w}d'] = df['close'].pct_change(w)
        feats[f'rsi_{w}'] = get_rsi(df['close'], w)
    
    for w in [20, 50, 200]:
        sma = df['close'].rolling(w).mean()
        feats[f'dist_sma_{w}'] = (df['close'] - sma) / sma
        
    feats['atr_norm'] = get_atr(df['high'], df['low'], df['close'], 14) / df['close']
    feats['vix_level'] = df['vix']
    feats['vix_roc_5'] = df['vix'].pct_change(5)
    
    for w in [10, 21]:
        feats[f'skew_{w}d'] = df['log_ret'].rolling(w).skew()
        feats[f'kurt_{w}d'] = df['log_ret'].rolling(w).kurt()

    for i in range(3): feats[f'hmm_prob_{i}'] = df[f'hmm_prob_{i}']

    # Target: 5D Forward Direction
    df['target_5D'] = (df['close'].shift(-5) > df['close']).astype(int)
    X_shifted = feats.shift(1)
    
    master_df = pd.concat([X_shifted, df[['target_5D', 'log_ret']]], axis=1).dropna()
    X = master_df.drop(columns=['target_5D', 'log_ret'])
    y = master_df['target_5D']

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing Asymmetric Walk-Forward Backtest...")
    
    lgb_params = {
        'objective': 'binary', 
        'learning_rate': 0.01, 
        'num_leaves': 12, 
        'max_depth': 4, 
        'feature_fraction': 0.7, 
        'verbosity': -1, 
        'random_state': 42
    }
    
    oof_predictions = pd.Series(index=X.index, dtype=float)
    
    for i in range(INITIAL_TRAIN_DAYS, len(X), STEP_DAYS):
        train_end = i
        test_end = min(i + STEP_DAYS, len(X))
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        
        # ASYMMETRIC TRAINING: Force the model to hate drawdowns.
        # We give a 2.0x weight to periods where the market fell (y=0). 
        # This makes the ML probability highly conservative.
        train_weights = np.where(y_train == 0, 2.0, 1.0)
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=250)
        model.fit(X_train, y_train, sample_weight=train_weights)
        
        oof_predictions.iloc[train_end:test_end] = model.predict_proba(X_test)[:, 1]

    # Align predictions with price data
    backtest_data = df.loc[oof_predictions.dropna().index].copy()
    backtest_data['prob_bull'] = oof_predictions.dropna()
    
    # =========================================================================
    # THE MAXIMUM CAPITAL PRESERVATION LOGIC
    # =========================================================================
    # 1. If HMM says we are in Regime 2 (High Volatility Panic) -> NEVER HOLD. 
    # 2. If HMM says Regime 0 or 1 -> Trust the conservative LightGBM model.
    # 3. If Model Conviction > 50% -> Go Long. Otherwise -> Cash.
    
    is_panic = backtest_data['hmm_prob_2'] > 0.50
    is_bullish = backtest_data['prob_bull'] > 0.50
    
    # np.where(condition, true_value, false_value)
    # If it is NOT a panic regime AND the model is bullish, take the trade.
    backtest_data['position'] = np.where((~is_panic) & is_bullish, 1, 0)
    
    # Calculate returns
    backtest_data['strat_ret'] = backtest_data['position'].shift(1) * backtest_data['log_ret']
    backtest_data['bnh_ret'] = backtest_data['log_ret']
    backtest_data = backtest_data.dropna()

    # Calculate and Print Metrics
    strat_metrics = calculate_metrics(backtest_data['strat_ret'], name="AI Regime Predictor (Asymmetric + Panic Block)")
    bnh_metrics = calculate_metrics(backtest_data['bnh_ret'], name="Buy & Hold (Nifty 50)")

    print("\n" + "="*80)
    print(" OUT-OF-SAMPLE BACKTEST RESULTS ".center(80, "="))
    print(f" Testing Period: {backtest_data.index[0].strftime('%Y-%m-%d')} to {backtest_data.index[-1].strftime('%Y-%m-%d')}")
    print("="*80)
    
    metrics_df = pd.DataFrame([bnh_metrics, strat_metrics]).set_index("Name")
    print(metrics_df.to_markdown())
    print("="*80)
    
    backtest_data['cum_strat'] = np.exp(backtest_data['strat_ret'].cumsum())
    backtest_data['cum_bnh'] = np.exp(backtest_data['bnh_ret'].cumsum())
    backtest_data[['cum_strat', 'cum_bnh']].to_csv("equity_curve.csv")
    print("\nEquity curve saved to equity_curve.csv.")

if __name__ == "__main__":
    main()
