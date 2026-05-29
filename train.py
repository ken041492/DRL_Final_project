import yaml
import joblib  # 新增 joblib 匯入
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.models.lstm_extractor import LSTMFeatureExtractor, DataPreprocessor, fetch_and_engineer_data
from src.env.trading_env import TradingEnv

def main():
    print("讀取設定檔...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    lookback = config.get("lookback", 30)
    hidden_size = config.get("hidden_size", 128)
    output_dim = config.get("output_dim", 64)
    
    settings = config.get("settings", {})
    stock_id = settings.get("stock_id", "0050")
    ticker = f"{stock_id}.TW"
    START_DATE = settings.get("start_date", "2025-01-27")
    END_DATE = settings.get("end_date", "2026-05-27")
    
    print(f"Loaded config: lookback={lookback}, hidden_size={hidden_size}, output_dim={output_dim}, ticker={ticker}, START_DATE={START_DATE}, END_DATE={END_DATE}")
    
    # 1. 資料收集與預處理
    # features_df, close_series = fetch_and_engineer_data(START_DATE, END_DATE, ticker=ticker, sentiment_csv_path="macro_sentiment_2021_2026_final.csv")
    features_df, close_series = fetch_and_engineer_data(START_DATE, END_DATE, ticker=ticker)
    print(f"取得歷史特徵資料形狀: {features_df.shape}")
    
    # 驗證特徵是否正確納入
    print(f"特徵欄位名稱: {list(features_df.columns)}")
    if "Daily_Sentiment" in features_df.columns:
        print("[OK] 成功納入新聞情緒分數 (Daily_Sentiment)！")
    else:
        print("[WARN] 警告：未找到新聞情緒分數 (Daily_Sentiment)！")
    
    preprocessor = DataPreprocessor(scaler_type='standard', lookback=lookback)
    scaled_features = preprocessor.fit_transform(features_df.values)
    
    # 2. 建立強化學習環境
    print("初始化 TradingEnv...")
    env = TradingEnv(df=scaled_features, close_prices=close_series, lookback=lookback)
    
    # 檢查環境是否符合 stable-baselines3 規範
    print("檢查環境規範...")
    check_env(env)
    print("環境檢查通過！")
    
    # 3. 實例化 PPO 模型
    print("建立 PPO 模型與自定義 LSTM 網路...")
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            hidden_size=hidden_size, 
            output_dim=output_dim
        )
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64
    )
    
    # 4. 開始訓練
    print("開始訓練 PPO 模型...")
    # 設定 total_timesteps，這裡預設為 50000 步
    model.learn(total_timesteps=50000)
    
    # 5. 儲存模型與預處理器 (Scaler)
    model_path = "ppo_macro_market_model"
    model.save(model_path)
    
    # 使用 joblib 儲存剛剛 fit 好的 preprocessor，以便測試時讀取
    joblib.dump(preprocessor, "ppo_macro_market_preprocessor.pkl")
    print(f"訓練完成，模型已儲存至 {model_path}.zip")
    print("預處理器 (Scaler) 已儲存至 ppo_macro_market_preprocessor.pkl")

if __name__ == "__main__":
    main()
