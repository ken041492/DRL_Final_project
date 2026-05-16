import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 設定字體以支援中文顯示 (Windows 預設為 Microsoft JhengHei)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

from src.models.lstm_extractor import DataPreprocessor, fetch_and_engineer_tsmc_data
from src.env.trading_env import TradingEnv

def main():
    print("讀取設定檔...")
    # 2. 讀取 configs/config.yaml 取得 lookback 等參數
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    lookback = config.get("lookback", 30)
    
    START_DATE = "2025-05-15"
    END_DATE = "2026-05-15"
    
    print("下載資料中...")
    features_df, close_series = fetch_and_engineer_tsmc_data(START_DATE, END_DATE)
    
    preprocessor = DataPreprocessor(scaler_type='standard', lookback=lookback)
    scaled_features = preprocessor.fit_transform(features_df.values)
    
    print("初始化環境...")
    env = TradingEnv(df=scaled_features, close_prices=close_series, lookback=lookback)
    
    print("載入模型...")
    model = PPO.load("ppo_tsmc_model")
    
    print("開始回測...")
    dates = []
    prices = []
    net_worths = []
    actions_list = []
    
    # 從 close_series 提取對應的日期與原始價格
    dates_array = close_series.index
    prices_array = close_series.values
    
    # 重置環境，取得初始狀態
    obs, info = env.reset()
    done = False
    initial_balance = env.initial_balance
    
    while not done:
        current_step = info['step']
        
        # 使用模型預測動作，設定 deterministic=True 保證測試穩定性
        action, _states = model.predict(obs, deterministic=True)
        
        # 從 array 中提取浮點數 (因為 action_space 改為 Box(shape=(1,)))
        act_value = float(action[0])
        
        # 紀錄執行動作當下的日期與價格
        dates.append(dates_array[current_step])
        prices.append(prices_array[current_step])
        actions_list.append(act_value)
        
        # 推進環境
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 紀錄執行動作後的淨值
        net_worths.append(info['net_worth'])
        
        done = terminated or truncated

    # 計算績效指標：總報酬率 與 最大回撤 (MDD)
    net_worth_array = np.array(net_worths)
    total_return = (net_worth_array[-1] - initial_balance) / initial_balance * 100
    
    # 計算 MDD
    peak = np.maximum.accumulate(net_worth_array)
    drawdown = (net_worth_array - peak) / peak
    mdd = drawdown.min() * 100
    
    print("-" * 30)
    print("回測完成！")
    print(f"總報酬率: {total_return:.2f}%")
    print(f"最大回撤 (MDD): {mdd:.2f}%")
    print("-" * 30)
    
    # 7. 繪製圖表
    print("繪製圖表中...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # --- 上半部：真實收盤價走勢與買賣點 ---
    ax1.plot(dates, prices, label="台積電收盤價", color="blue", alpha=0.6)
    
    # 篩選出買進與賣出的座標 (依據新的連續動作空間：> 0.01 為買進，< -0.01 為賣出)
    buy_dates = [dates[i] for i, a in enumerate(actions_list) if a > 0.01]
    buy_prices = [prices[i] for i, a in enumerate(actions_list) if a > 0.01]
    
    sell_dates = [dates[i] for i, a in enumerate(actions_list) if a < -0.01]
    sell_prices = [prices[i] for i, a in enumerate(actions_list) if a < -0.01]
    
    # 在圖表標示紅點(買進)與綠點(賣出)
    ax1.scatter(buy_dates, buy_prices, marker="^", color="red", label="買進 (Action > 0)", s=100, zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker="v", color="green", label="賣出 (Action < 0)", s=100, zorder=5)
    
    ax1.set_title("台積電 (2330.TW) 收盤價與 AI 交易標示", fontsize=16)
    ax1.set_ylabel("價格 (TWD)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # --- 下半部：帳戶淨值變化 ---
    ax2.plot(dates, net_worths, label="帳戶淨值 (Net Worth)", color="purple", linewidth=2)
    ax2.set_title(f"帳戶淨值變化 (總報酬: {total_return:.2f}%, MDD: {mdd:.2f}%)", fontsize=16)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.set_ylabel("淨值 (TWD)", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 儲存圖片
    plt.savefig("evaluate_result.png")
    print("圖表已儲存為 evaluate_result.png")
    
    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    main()
