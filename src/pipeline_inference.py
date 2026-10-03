import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

class NiftyRegimePipeline:
    def __init__(self, db_path="data/market_data.sqlite"):
        self.ticker = "^NSEI"
        self.db_path = db_path
        
    def fetch_and_update_data(self):
        """Fetches Nifty 50 delta and maintains SQLite database integrity."""
        print(f"Fetching {self.ticker} data...")
        raw_data = yf.download(self.ticker, period="5y", interval="1d", progress=False)
        raw_data.columns = [col[0] for col in raw_data.columns] # Flatten MultiIndex
        
        # Forward-fill missing data to handle Muhurat/irregular sessions
        df = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().dropna()
        
        # Persist to SQLite
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("nifty50", conn, if_exists="replace", index=True)
        return df

    def compute_yang_zhang_volatility(self, df, window=20):
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
        
        df['YZ_Vol'] = np.sqrt(yz_var) * np.sqrt(252) # Annualized
        return df

    def triple_barrier_labels(self, df, vol_window=20, pt_sl=[1, 2], horizon=10):
        """Path-aware labeling: [1] Hit Take Profit, [-1] Hit Stop Loss, [0] Time Expired."""
        df = self.compute_yang_zhang_volatility(df, vol_window)
        events = pd.DataFrame(index=df.index)
        
        # Daily volatility target
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
            
            # Find which barrier is hit first
            hit_upper = path[path >= upper].index.min()
            hit_lower = path[path <= lower].index.min()
            
            if pd.isna(hit_upper) and pd.isna(hit_lower):
                labels[timestamp] = 0 # Time expiration
            elif pd.isna(hit_lower):
                labels[timestamp] = 1 # Profit
            elif pd.isna(hit_upper):
                labels[timestamp] = -1 # Stop Loss
            else:
                labels[timestamp] = 1 if hit_upper < hit_lower else -1
                
        df['Target_Label'] = labels
        return df

    def detect_market_regime(self, df):
        """Uses Gaussian Mixture HMM to output continuous state probabilities."""
        df['Returns'] = df['Close'].pct_change()
        clean_df = df.dropna()
        
        # Features for HMM: Returns and Yang-Zhang Volatility
        X = clean_df[['Returns', 'YZ_Vol']].values
        
        hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
        hmm_model.fit(X)
        
        # Get posterior probabilities of each regime
        regime_probs = hmm_model.predict_proba(X)
        clean_df['Prob_Regime_1'] = regime_probs[:, 0]
        clean_df['Prob_Regime_2'] = regime_probs[:, 1]
        clean_df['Prob_Regime_3'] = regime_probs[:, 2]
        
        return clean_df

    def meta_labeling_sizing(self, df):
        """XGBoost secondary classifier to determine execution size/rejection."""
        features = ['YZ_Vol', 'Prob_Regime_1', 'Prob_Regime_2', 'Prob_Regime_3']
        X = df[features].iloc[:-10] # Exclude unresolved recent barriers
        y = (df['Target_Label'].iloc[:-10] == 1).astype(int) # 1 if primary succeeded, 0 otherwise
        
        meta_model = XGBClassifier(eval_metric='logloss', objective='binary:logistic')
        meta_model.fit(X, y)
        
        # Predict probability of success for today
        latest_features = df[features].iloc[[-1]]
        prob_success = meta_model.predict_proba(latest_features)[0][1]
        
        return prob_success

    def run_daily_inference(self):
        """Executes the full pipeline and saves artifacts."""
        df = self.fetch_and_update_data()
        df = self.triple_barrier_labels(df)
        df = self.detect_market_regime(df)
        
        # DRL / Meta-Labeling Output
        prob_success = self.meta_labeling_sizing(df)
        latest_close = float(df['Close'].iloc[-1])
        
        # Determine sizing via calibration threshold
        allocation_weight = float(prob_success) if prob_success > 0.55 else 0.0
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "latest_close": latest_close,
            "meta_probability_success": float(prob_success),
            "target_allocation_weight": allocation_weight,
            "regime_probabilities": {
                "r1": float(df['Prob_Regime_1'].iloc[-1]),
                "r2": float(df['Prob_Regime_2'].iloc[-1]),
                "r3": float(df['Prob_Regime_3'].iloc[-1])
            }
        }
        
        with open("results/daily_inference.json", "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Inference complete. Target Allocation: {allocation_weight*100:.1f}%")

if __name__ == "__main__":
    pipeline = NiftyRegimePipeline()
    pipeline.run_daily_inference()
