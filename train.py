import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.models.lstm_extractor import LSTMFeatureExtractor, DataPreprocessor, fetch_and_engineer_tsmc_data
from src.env.trading_env import TradingEnv

def main():
    print("讀取設定檔...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    lookback = config.get("lookback", 30)
    hidden_size = config.get("hidden_size", 128)
    output_dim = config.get("output_dim", 64)
    
    print(f"Loaded config: lookback={lookback}, hidden_size={hidden_size}, output_dim={output_dim}")
    
    # 1. 資料收集與預處理
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    
    features_df, close_series = fetch_and_engineer_tsmc_data(START_DATE, END_DATE)
    print(f"取得歷史特徵資料形狀: {features_df.shape}")
    
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
    
    # 5. 儲存模型
    model_path = "ppo_tsmc_model"
    model.save(model_path)
    print(f"訓練完成，模型已儲存至 {model_path}.zip")

if __name__ == "__main__":
    main()
