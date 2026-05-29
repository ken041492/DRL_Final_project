"""
LSTM 特徵萃取器模組 (LSTM Feature Extractor Module)

此模組提供深度強化學習 (DRL) 環境所需的狀態 (State) 特徵萃取功能。
包含：
1. 資料收集與特徵工程 (Data Fetcher): 使用 yfinance 下載台積電資料，並計算技術指標。
2. 資料預處理 (Data Preprocessor): 負責特徵縮放 (Scaling) 與滑動窗口 (Sliding Window) 切割。
3. LSTM 特徵萃取網路 (LSTM Feature Extractor): 將時間序列資料轉換為固定維度的隱藏狀態向量，供 PPO 的 Actor-Critic 網路使用。
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def fetch_and_engineer_data(start_date, end_date, ticker=None, sentiment_csv_path=None):
    """
    下載股票歷史資料並計算技術指標 (SMA, RSI, MACD, logret)
    ，並可選擇性地融合 LLM 情緒分數特徵。
    
    參數:
        start_date (str): 開始日期，例如 "2020-01-01"
        end_date (str): 結束日期，例如 "2024-12-31"
        ticker (str, optional): 股票代號 (預設為 None，若為 None 則預設為 "2330.TW")
        sentiment_csv_path (str, optional): 情緒分數 CSV 的檔案路徑 (預設為 None)
        
    回傳:
        features (pd.DataFrame): 處理好並清除 NaN 的特徵 DataFrame
        close_prices (pd.Series): 對應的原始 Close 價格 Series (供 RL 環境結算報酬用)
    """
    if ticker is None:
        ticker = "2330.TW"
        
    print(f"正在下載 {ticker} 資料 ({start_date} 至 {end_date})...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # 針對 yfinance 可能回傳 MultiIndex 的處理 (防止舊版/新版相容問題)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    print("計算技術指標中...")
    # SMA 計算
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # RSI 計算
    delta = df['Close'].diff()
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD 計算
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # logret (對數報酬率) 計算
    df['logret'] = np.log(df['Close'].shift(-1)) - np.log(df['Close'])
    
    # 下載 0050.TW 資料並計算相對強弱指標
    print(f"正在下載 0050.TW 資料 ({start_date} 至 {end_date})...")
    df_0050 = yf.download("0050.TW", start=start_date, end=end_date)
    if isinstance(df_0050.columns, pd.MultiIndex):
        df_0050.columns = df_0050.columns.droplevel(1)
        
    df_0050['logret_0050'] = np.log(df_0050['Close'].shift(-1)) - np.log(df_0050['Close'])
    
    # 合併 0050.TW 的對數報酬率到 df，使用 left join 確保對齊目標 ticker 交易日
    df = df.join(df_0050[['logret_0050']], how='left')
    df['logret_0050'] = df['logret_0050'].ffill().fillna(0.0)
    
    # 計算 Relative_Strength = ticker_logret - 0050_logret
    df['Relative_Strength'] = df['logret'] - df['logret_0050']
    
    # 建立 100% 定常化 (Stationary) 特徵，移除絕對價格依賴
    df['Close_to_SMA5'] = df['Close'] / df['SMA_5'] - 1
    df['Close_to_SMA20'] = df['Close'] / df['SMA_20'] - 1
    df['High_to_Low'] = df['High'] / df['Low'] - 1
    df['Close_to_Open'] = df['Close'] / df['Open'] - 1
    
    df['MACD_norm'] = df['MACD'] / df['Close']
    df['MACD_SIGNAL_norm'] = df['MACD_SIGNAL'] / df['Close']
    df['MACD_HIST_norm'] = df['MACD_HIST'] / df['Close']
    
    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 若有提供情緒分數檔案，則進行融合
    if sentiment_csv_path is not None:
        import os
        if os.path.exists(sentiment_csv_path):
            print("正在融合情緒分數特徵...")
            sentiment_df = pd.read_csv(sentiment_csv_path)
            
            # 1. 時間聚合：將 date 欄位轉換為 YYYY-MM-DD 格式
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.strftime('%Y-%m-%d')
            
            # 將同一天的新聞分數取平均值 (Daily_Sentiment)，並計算當天新聞數量 (News_Count)
            daily_sentiment = sentiment_df.groupby('date').agg(
                Daily_Sentiment=('sentiment_score', 'mean'),
                News_Count=('sentiment_score', 'count')
            ).reset_index()
            
            # 轉換為 DatetimeIndex 以便與 yfinance 的索引對齊
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment.set_index('date', inplace=True)
            
            # 2. Left Join 融合：將這份每日情緒表與股票價量表透過日期進行 Left Join
            # yfinance 回傳的 df 索引為 DatetimeIndex，可以直接與 daily_sentiment join
            df = df.join(daily_sentiment, how='left')
            
            # 3. 處理空缺值 (Sparse Data)：將沒有新聞的交易日補 0.0 與 0
            df['Daily_Sentiment'] = df['Daily_Sentiment'].ffill()
            df['Daily_Sentiment'] = df['Daily_Sentiment'].fillna(0.0)
            df['News_Count'] = df['News_Count'].fillna(0).astype(int)

            # 計算 5 日移動平均情緒分數 (Sentiment_SMA5)，用 ffill 處理 rolling 產生的 NaN，再補 0.0
            df['Sentiment_SMA5'] = df['Daily_Sentiment'].rolling(5).mean().ffill().fillna(0.0)

            # 4. (進階技巧) 檢查是否有 Inf 值
            # 有時候算平均值會產生極端數值，確保資料乾淨
            df.replace([np.inf, -np.inf], 0.0, inplace=True)
        else:
            print(f"警告: 找不到情緒分數檔案 {sentiment_csv_path}，將略過情緒特徵融合。")

    # 清除包含 NaN 值的行 (例如 rolling/diff 造成的空缺)
    df.dropna(inplace=True)

    # 選擇 100% 定常化 (Stationary) 的特徵作為模型輸入，徹底避免價格水位變化造成的外推失敗
    feature_cols = [
        'RSI_14', 'logret', 'Close_to_SMA5', 'Close_to_SMA20', 
        'High_to_Low', 'Close_to_Open', 'MACD_norm', 
        'MACD_SIGNAL_norm', 'MACD_HIST_norm', 'Volume_ratio',
        'Relative_Strength'
    ]
    
    # 若有成功融合情緒特徵，則加入
    if 'Daily_Sentiment' in df.columns:
        feature_cols.extend(['Daily_Sentiment', 'Sentiment_SMA5', 'News_Count'])
        
    features = df[feature_cols].copy()
    close_prices = df['Close'].copy()
    
    return features, close_prices


class DataPreprocessor:
    """
    資料預處理器，負責處理特徵縮放與滑動窗口資料集建立。
    此架構保留擴充性，未來可輕易併入「大盤總經數據」與「LLM 新聞情緒分數」。
    """
    def __init__(self, scaler_type='standard', lookback=30):
        """
        初始化資料預處理器
        
        參數:
            scaler_type (str): 'standard' 或 'minmax'，決定縮放器種類
            lookback (int): 時間序列滑動窗口的長度 (即考慮過去 N 天的資料)
        """
        self.lookback = lookback
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type 必須為 'standard' 或 'minmax'")
            
    def fit_transform(self, df_features):
        """
        適配並縮放特徵資料 (用於訓練集)
        
        參數:
            df_features (pd.DataFrame or np.ndarray): 原始特徵資料
            
        回傳:
            np.ndarray: 縮放後的特徵資料
        """
        return self.scaler.fit_transform(df_features)
        
    def transform(self, df_features):
        """
        僅縮放特徵資料 (用於驗證/測試集或實際推論環節)
        
        參數:
            df_features (pd.DataFrame or np.ndarray): 原始特徵資料
            
        回傳:
            np.ndarray: 縮放後的特徵資料
        """
        return self.scaler.transform(df_features)
        
    def create_sliding_window(self, scaled_features):
        """
        將縮放後的 2D 序列轉換為 3D 的滑動窗口資料 (Batch, Lookback, Features)
        
        參數:
            scaled_features (np.ndarray): 已經過縮放的特徵矩陣
            
        回傳:
            np.ndarray: 形狀為 (N - lookback, lookback, num_features) 的特徵矩陣
        """
        X = []
        for i in range(len(scaled_features) - self.lookback):
            X.append(scaled_features[i : i + self.lookback])
        return np.array(X)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM 特徵萃取網路，專為 PPO 演算法設計的狀態 (State) 萃取器。
    接收過去 N 天的 OHLCV (及未來擴充) 序列特徵，輸出一個固定維度的濃縮隱藏狀態向量。
    """
    def __init__(self, observation_space: gym.spaces.Box, hidden_size=128, output_dim=64, num_layers=1, batch_first=True):
        """
        初始化 LSTM 特徵萃取器
        
        參數:
            observation_space (gym.spaces.Box): 來自 gymnasium 環境的觀察空間
            hidden_size (int): LSTM 隱藏層的維度大小
            output_dim (int): 最終輸出的特徵向量維度 (作為 PPO Actor/Critic 的狀態輸入)
            num_layers (int): LSTM 層數
            batch_first (bool): 輸入的形狀是否為 (Batch, Seq, Feature)
        """
        # 呼叫 BaseFeaturesExtractor 的初始化，必須傳入 observation_space 與 features_dim
        super().__init__(observation_space, features_dim=output_dim)
        
        # 從 observation_space 取得輸入特徵的維度 (lookback, num_features)
        # 此處 observation_space.shape 為 (lookback, num_features)，因此取 shape[1] 作為 input_size
        input_size = observation_space.shape[1]
        
        # LSTM 主體
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )
        
        # Layer Normalization 幫助訓練穩定
        self.norm = nn.LayerNorm(hidden_size)
        
        # 線性映射層，將 LSTM 最後一個時間步的隱藏狀態轉換為指定維度的特徵向量
        self.fc = nn.Linear(hidden_size, output_dim)
        
        # 激活函數，可選用 ReLU 或 Tanh 根據後續 Actor-Critic 設計
        self.activation = nn.ReLU()
        
    def forward(self, observations: torch.Tensor):
        """
        前向傳播
        
        參數:
            observations (torch.Tensor): 輸入張量，形狀為 (Batch Size, Lookback, Input Features)
            
        回傳:
            torch.Tensor: 濃縮後的特徵向量，形狀為 (Batch Size, Output Dim)
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        # _ : (h_n, c_n) 包含最後一個時間步的 hidden state 與 cell state
        lstm_out, _ = self.lstm(observations)
        
        # 只取最後一個時間步 (Last time step) 的輸出作為序列特徵代表
        last_step_out = lstm_out[:, -1, :]
        
        # 正規化 (Layer Normalization)
        normalized_out = self.norm(last_step_out)
        
        # 透過線性層壓縮至指定的特徵維度，再經過 ReLU 處理
        feature_vector = self.activation(self.fc(normalized_out))
        
        return feature_vector
