import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.models.lstm_extractor import LSTMFeatureExtractor, DataPreprocessor, fetch_and_engineer_data
from src.env.trading_env import TradingEnv

# 設定字體以支援中文顯示 (Windows 預設為 Microsoft JhengHei)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def run_training_and_eval(use_sentiment, config):
    lookback = config.get("lookback", 30)
    hidden_size = config.get("hidden_size", 128)
    output_dim = config.get("output_dim", 64)
    
    settings = config.get("settings", {})
    stock_id = settings.get("stock_id", "0050")
    ticker = f"{stock_id}.TW"
    
    # 訓練時間區間
    TRAIN_START = settings.get("start_date", "2021-01-01")
    TRAIN_END = settings.get("end_date", "2024-12-31")
    
    # 測試時間區間
    TEST_START = "2025-01-01"
    TEST_END = "2026-05-26"
    
    sentiment_path = "data/macro_sentiment_2021_2026_final.csv" if use_sentiment else None
    tag = "有新聞分數" if use_sentiment else "無新聞分數"
    model_name = "models/ppo_with_sentiment_model" if use_sentiment else "models/ppo_without_sentiment_model"
    preprocessor_name = "models/preprocessor_with_sentiment.pkl" if use_sentiment else "models/preprocessor_without_sentiment.pkl"
    
    print(f"\n{'='*50}")
    print(f"開始執行實驗: {tag}")
    print(f"{'='*50}")
    
    # === 1. 訓練階段 ===
    print("[訓練階段] 下載與預處理資料...")
    train_features_df, train_close_series = fetch_and_engineer_data(
        TRAIN_START, TRAIN_END, ticker=ticker, sentiment_csv_path=sentiment_path
    )
    
    preprocessor = DataPreprocessor(scaler_type='standard', lookback=lookback)
    train_scaled = preprocessor.fit_transform(train_features_df.values)
    
    train_env = TradingEnv(df=train_scaled, close_prices=train_close_series, lookback=lookback)
    
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(hidden_size=hidden_size, output_dim=output_dim)
    )
    
    model = PPO("MlpPolicy", env=train_env, policy_kwargs=policy_kwargs, verbose=0, learning_rate=3e-4, batch_size=64)
    
    print("[訓練階段] 開始訓練模型 (50000 steps)...")
    model.learn(total_timesteps=50000)
    
    # 儲存模型與預處理器
    model.save(model_name)
    joblib.dump(preprocessor, preprocessor_name)
    
    # === 2. 測試階段 ===
    print("[測試階段] 下載與預處理測試資料...")
    test_features_df, test_close_series = fetch_and_engineer_data(
        TEST_START, TEST_END, ticker=ticker, sentiment_csv_path=sentiment_path
    )
    
    # 使用訓練階段配適好的 preprocessor 進行 transform
    test_scaled = preprocessor.transform(test_features_df.values)
    test_env = TradingEnv(df=test_scaled, close_prices=test_close_series, lookback=lookback)
    
    print("[測試階段] 開始回測...")
    obs, info = test_env.reset()
    done = False
    
    dates_array = test_close_series.index
    prices_array = test_close_series.values
    
    dates = []
    net_worths = []
    actions_list = []
    prices = []
    
    initial_balance = test_env.initial_balance
    
    while not done:
        current_step = info['step']
        dates.append(dates_array[current_step])
        prices.append(prices_array[current_step])
        
        action, _states = model.predict(obs, deterministic=True)
        act_value = float(action[0])
        actions_list.append(act_value)
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        net_worths.append(info['net_worth'])
        done = terminated or truncated
        
    net_worth_array = np.array(net_worths)
    total_return = (net_worth_array[-1] - initial_balance) / initial_balance * 100
    
    peak = np.maximum.accumulate(net_worth_array)
    drawdown = (net_worth_array - peak) / peak
    mdd = drawdown.min() * 100
    
    print(f"實驗 [{tag}] 回測完成:")
    print(f"總報酬率: {total_return:.2f}%")
    print(f"最大回撤 (MDD): {mdd:.2f}%")
    
    return {
        "dates": dates,
        "prices": prices,
        "actions": actions_list,
        "net_worths": net_worths,
        "total_return": total_return,
        "mdd": mdd,
        "tag": tag
    }

def main():
    print("讀取設定檔...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    stock_id = config.get("settings", {}).get("stock_id", "0050")
        
    # 執行兩組實驗
    res_with = run_training_and_eval(True, config)
    res_without = run_training_and_eval(False, config)
    
    # 繪圖比較
    print("\n繪製比較圖表...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    
    dates_with = res_with["dates"]
    dates_without = res_without["dates"]
    
    def plot_actions(ax, res_dict, title):
        dates = res_dict["dates"]
        prices = res_dict["prices"]
        actions = res_dict["actions"]
        
        ax.plot(dates, prices, label=f"{stock_id} 收盤價", color="blue", alpha=0.5)
        
        buy_dates = [dates[i] for i, a in enumerate(actions) if a > 0.01]
        buy_prices = [prices[i] for i, a in enumerate(actions) if a > 0.01]
        
        sell_dates = [dates[i] for i, a in enumerate(actions) if a < -0.01]
        sell_prices = [prices[i] for i, a in enumerate(actions) if a < -0.01]
        
        ax.scatter(buy_dates, buy_prices, marker="^", color="red", label="買進 (Action > 0)", s=80, zorder=5)
        ax.scatter(sell_dates, sell_prices, marker="v", color="green", label="賣出 (Action < 0)", s=80, zorder=5)
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("價格 (TWD)", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    # 上半部：有新聞分數的交易動作
    plot_actions(ax1, res_with, f"AI 交易標示 - {res_with['tag']}")
    
    # 中間部：無新聞分數的交易動作
    plot_actions(ax2, res_without, f"AI 交易標示 - {res_without['tag']}")
    
    # 下半部：帳戶淨值比較
    ax3.plot(dates_with, res_with["net_worths"], label=f'{res_with["tag"]} (報酬: {res_with["total_return"]:.2f}%, MDD: {res_with["mdd"]:.2f}%)', color='red', linewidth=2)
    ax3.plot(dates_without, res_without["net_worths"], label=f'{res_without["tag"]} (報酬: {res_without["total_return"]:.2f}%, MDD: {res_without["mdd"]:.2f}%)', color='blue', linewidth=2, linestyle='--')
    
    ax3.set_title(f"{stock_id} - 帳戶淨值比較 (有無加入新聞情緒分數)", fontsize=14)
    ax3.set_xlabel("日期", fontsize=12)
    ax3.set_ylabel("帳戶淨值 (TWD)", fontsize=12)
    ax3.legend(loc="upper left", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    save_path = f"compare_sentiment_result_{stock_id}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"圖表已儲存為 {save_path}")

if __name__ == "__main__":
    main()
