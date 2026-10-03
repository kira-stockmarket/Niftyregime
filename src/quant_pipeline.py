import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
import os
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, brier_score_loss, classification_report
import warnings
warnings.filterwarnings('ignore')

class InstitutionalQuantModel:
    def __init__(self, ticker="^NSEI", db_path="data/market_data.sqlite"):
        self.ticker = ticker
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
    def fetch_data(self):
        """Fetches delta and maintains SQLite database integrity."""
        raw_data = yf.download(self.ticker, period="10y", interval="1d", progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = [col[0] for col in raw_data.columns]
        
        df = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().dropna()
        df['Returns'] = df['Close'].pct_change()
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(f"market_data_{self.ticker.replace('^', '')}", conn, if_exists="replace", index=True)
        return df.dropna()

    def yang_zhang_volatility(self, df, window=20):
        """Calculates the minimum-variance, unbiased Yang-Zhang volatility."""
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_co = np.log(df['Close'] / df['Open'])
        log_oc = np.log(df['Open'] / df['Close'].shift(1))
        log_cc = np.log(df['Close'] / df['Close'].shift(1))
        
        rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
        close_var = log_cc.rolling(window).var()
        open_var = log_oc.rolling(window).var()
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = open_var + k * close_var + (1 - k) * rs_var
        
        df['YZ_Vol'] = np.sqrt(yz_var) * np.sqrt(252)
        return df

    def triple_barrier_labels(self, df, vol_window=20, pt_sl=[1, 2], horizon=10):
        """Path-aware labeling for realistic execution modeling."""
        df = self.yang_zhang_volatility(df, vol_window)
        events = pd.DataFrame(index=df.index)
        daily_vol = df['YZ_Vol'] / np.sqrt(252)
        
        events['t1'] = df.index.to_series().shift(-horizon)
        events['Upper_Barrier'] = df['Close'] * (1 + (daily_vol * pt_sl[0]))
        events['Lower_Barrier'] = df['Close'] * (1 - (daily_vol * pt_sl[1]))
        
        labels = pd.Series(0, index=df.index)
        
        for loc, timestamp in enumerate(df.index):
            if pd.isna(events.loc[timestamp, 't1']):
                continue
            
            path = df['Close'].iloc[loc : loc + horizon]
            upper = events.loc[timestamp, 'Upper_Barrier']
            lower = events.loc[timestamp, 'Lower_Barrier']
            
            hit_upper = path[path >= upper].index.min()
            hit_lower = path[path <= lower].index.min()
            
            if pd.isna(hit_upper) and pd.isna(hit_lower):
                labels[timestamp] = 0
            elif pd.isna(hit_lower):
                labels[timestamp] = 1
            elif pd.isna(hit_upper):
                labels[timestamp] = -1
            else:
                labels[timestamp] = 1 if hit_upper < hit_lower else -1
                
        df['Target_Label'] = labels
        df['Target_Binary'] = (df['Target_Label'] == 1).astype(int)
        return df.dropna()

    def detect_regimes(self, df):
        """Gaussian Mixture HMM for continuous state probabilities."""
        X = df[['Returns', 'YZ_Vol']].values
        hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        hmm_model.fit(X)
        
        regime_probs = hmm_model.predict_proba(X)
        for i in range(3):
            df[f'Prob_Regime_{i+1}'] = regime_probs[:, i]
        return df

    def run_daily_inference(self):
        """Executes full pipeline for the current day."""
        print(f"Running daily inference for {self.ticker}...")
        df = self.fetch_data()
        df = self.triple_barrier_labels(df)
        df = self.detect_regimes(df)
        
        features = ['YZ_Vol', 'Prob_Regime_1', 'Prob_Regime_2', 'Prob_Regime_3']
        X = df[features].iloc[:-10]
        y = df['Target_Binary'].iloc[:-10]
        
        meta_model = XGBClassifier(eval_metric='logloss', objective='binary:logistic', max_depth=3)
        meta_model.fit(X, y)
        
        latest_features = df[features].iloc[[-1]]
        prob_success = float(meta_model.predict_proba(latest_features)[0][1])
        allocation_weight = prob_success if prob_success > 0.55 else 0.0
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "ticker": self.ticker,
            "latest_close": float(df['Close'].iloc[-1]),
            "meta_probability_success": prob_success,
            "target_allocation_weight": allocation_weight,
            "regime_probabilities": {
                "r1": float(df['Prob_Regime_1'].iloc[-1]),
                "r2": float(df['Prob_Regime_2'].iloc[-1]),
                "r3": float(df['Prob_Regime_3'].iloc[-1])
            }
        }
        
        with open(f"results/inference_{self.ticker.replace('^', '')}.json", "w") as f:
            json.dump(results, f, indent=4)
        print(json.dumps(results, indent=4))

    def run_backtest(self, train_days=1000, test_days=250, purge_days=20):
        """Purged Walk-Forward CV to prevent look-ahead bias."""
        print(f"Running institutional backtest for {self.ticker}...")
        df = self.fetch_data()
        df = self.triple_barrier_labels(df)
        df = self.detect_regimes(df)
        
        features = ['YZ_Vol', 'Prob_Regime_1', 'Prob_Regime_2', 'Prob_Regime_3']
        X = df[features]
        y = df['Target_Binary']
        
        oos_preds, oos_probs = pd.Series(dtype=float), pd.Series(dtype=float)
        
        for start_idx in range(0, len(df) - train_days - test_days, test_days):
            train_end = start_idx + train_days
            test_start = train_end + purge_days
            test_end = test_start + test_days
            
            if test_end > len(df): break
                
            X_train, y_train = X.iloc[start_idx:train_end], y.iloc[start_idx:train_end]
            X_test = X.iloc[test_start:test_end]
            
            model = XGBClassifier(eval_metric='logloss', objective='binary:logistic', max_depth=3)
            model.fit(X_train, y_train)
            
            probs = model.predict_proba(X_test)[:, 1]
            oos_probs = pd.concat([oos_probs, pd.Series(probs, index=X_test.index)])
            oos_preds = pd.concat([oos_preds, pd.Series((probs > 0.55).astype(int), index=X_test.index)])

        aligned = pd.DataFrame({'prob': oos_probs, 'pred': oos_preds, 'actual': y}).dropna()
        
        brier = brier_score_loss(aligned['actual'], aligned['prob'])
        acc = accuracy_score(aligned['actual'], aligned['pred'])
        
        strategy_returns = df.loc[aligned.index, 'Returns'] * aligned['pred']
        ann_return = strategy_returns.mean() * 252
        ann_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Global Accuracy:       {acc:.2%}")
        print(f"Brier Score (Risk):    {brier:.4f}")
        print(f"Annualized Return:     {ann_return:.2%}")
        print(f"Annualized Volatility: {ann_vol:.2%}")
        print(f"Out-of-Sample Sharpe:  {sharpe:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inference', 'backtest'], default='inference')
    parser.add_argument('--ticker', default='^NSEI', help='Ticker symbol (e.g., ^NSEI, INFY.NS, ASIANPAINT.NS)')
    args = parser.parse_args()
    
    model = InstitutionalQuantModel(ticker=args.ticker)
    if args.mode == 'inference':
        model.run_daily_inference()
    else:
        model.run_backtest()
