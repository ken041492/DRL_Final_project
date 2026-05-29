import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    自定義交易環境，繼承自 gymnasium.Env
    提供給深度強化學習 (DRL) 代理人進行股票交易訓練
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, close_prices, lookback=30):
        """
        初始化交易環境
        
        參數:
            df (pd.DataFrame or np.ndarray): 已經處理好的特徵 DataFrame (包含縮放後的技術指標)
            close_prices (pd.Series or np.ndarray): 對應的 Close 價格，用於計算真實交易盈虧
            lookback (int): 觀察窗口大小 (過去 N 天的資料)
        """
        super(TradingEnv, self).__init__()
        
        self.lookback = lookback
        
        # 確保資料轉為 numpy array 格式，加速後續切片存取
        if isinstance(df, pd.DataFrame):
            self.df = df.values
        else:
            self.df = np.array(df)
            
        if isinstance(close_prices, (pd.Series, pd.DataFrame)):
            self.close_prices = close_prices.values.flatten()
        else:
            self.close_prices = np.array(close_prices).flatten()
            
        self.num_features = self.df.shape[1]
        
        # 動作空間： -1 (賣出), 0 (持有), 1 (買進)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 觀察空間: Box，形狀為 (lookback, num_features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback, self.num_features), 
            dtype=np.float32
        )
        
        # 初始帳戶設定
        self.initial_balance = 100000.0  # 初始資金 10 萬
        self.fee_rate = 0.002            # 包含手續費與交易稅約略 0.2%
        self.action_penalty_lambda = 50.0 # 動作變化懲罰係數
        
        # 定義狀態變數
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance  # 追蹤歷史最高淨值
        self.last_action = 0.0           # 紀錄上一步的動作

    def reset(self, seed=None, options=None):
        """
        重置環境狀態，回到初始設定
        """
        super().reset(seed=seed)
        
        # 重置時間步與帳戶狀態
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance  # 重置歷史最高淨值
        self.last_action = 0.0           # 重置上一步動作
        
        # 取得初始觀察值 (過去 lookback 天的特徵)
        obs = self.df[self.current_step - self.lookback : self.current_step]
        
        # 紀錄額外資訊
        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held
        }
        
        return np.array(obs, dtype=np.float32), info

    @property
    def position(self):
        """
        定義 position 屬性，當持股數大於 0 時返回 1，空倉時返回 0
        """
        return 1 if self.shares_held > 0.0 else 0

    def step(self, action):
        """
        執行動作並推展環境
        
        參數:
            action (int): 動作代碼 (0: 賣出, 1: 持有, 2: 買進)
        """
        # 確保數值嚴格在範圍內
        act = float(np.clip(action[0], -1.0, 1.0))
        
        # 新增「交易死區（Deadzone Threshold）」
        # 如果動作絕對值小於 0.15，強制設為 0 (持有/不動)，避免頻繁微調
        if abs(act) < 0.15:
            act = 0.0

        # 1. 取得當前執行動作的價格與昨收價格，計算純資產漲跌幅
        prev_price = self.close_prices[self.current_step - 1]
        current_price = self.close_prices[self.current_step]
        asset_return = (current_price - prev_price) / prev_price
        
        # 紀錄執行動作前的淨值與部位狀態，作為獎勵判定基準
        self.prev_net_worth = self.balance + self.shares_held * prev_price
        had_position = self.position > 0

        # 執行動作與紀錄是否有交易發生
        threshold = 0.3
        transaction_happened = False
        
        if act > threshold:  # 強烈看多：滿倉買進 / 保持持倉
            # 滿倉買進：如果手上有現金，就全額買進
            if self.balance > 0.0:
                # 扣除手續費後的實際投入金額
                invest_amount = self.balance / (1 + self.fee_rate)
                shares_bought = invest_amount / current_price
                if shares_bought > 0.0:
                    self.shares_held += shares_bought
                    self.balance -= (invest_amount * (1 + self.fee_rate))
                    transaction_happened = True
            
        elif act < -threshold:  # 強烈看空：清倉賣出 / 保持空手
            # 清倉賣出：如果手上有股票，就全額賣出
            if self.shares_held > 0.0:
                revenue = self.shares_held * current_price
                transaction_fee = revenue * self.fee_rate
                self.balance += (revenue - transaction_fee)
                self.shares_held = 0.0
                transaction_happened = True
                
        # 如果在 [-threshold, threshold] 之間，執行維持現狀 (transaction_happened = False，無任何買賣或手續費扣除)
            
        # 4. 帳戶淨值更新：如實扣除真實手續費以供最終淨值回測
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 2. 基礎獎勵設計 (業界標準部位報酬法)
        if had_position:
            reward = asset_return * 100.0  # 漲就給正分，跌就給負分
        else:
            reward = 0.0  # 空手則無部位報酬
            
        # 3. 獨立的交易成本摩擦懲罰
        if transaction_happened:
            reward -= self.fee_rate * 100.0
            
        # 更新上一步動作
        self.last_action = act
        
        # 推進時間步
        self.current_step += 1
        
        # 判斷是否結束
        terminated = bool(self.current_step >= len(self.df) or \
                          self.net_worth < self.initial_balance * 0.1)
        truncated = False
        
        # 取得下一個狀態的觀察值（優雅的寫法）
        obs = self.df[self.current_step - self.lookback : self.current_step] \
              if not terminated else self.df[-self.lookback:]
        
        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,     # 總資產（戰力指標）
            "balance": self.balance,         # 剩餘現金（看它手上有沒有子彈）
            "shares_held": self.shares_held, # 持股數量（看它是不是滿倉）
            "action_taken": act              # AI 這次出的力道（-1.0 ~ 1.0）
        }
        
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        """
        預留的渲染方法，可於此處印出當前帳戶狀態
        """
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.4f}")
