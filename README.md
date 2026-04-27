# 穿越極端市場的風險感知系統：基於 PPO + LSTM 架構之動態資產配置與 2026 壓力測試 
> Risk-Aware Dynamic Asset Allocation: A Hybrid PPO-LSTM Framework under Extreme Market Stress

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![Status](https://img.shields.io/badge/Status-Final_Project-orange)

## 👥 專案成員 (Team Members)
* **張凱翔** (國立中興大學 資訊管理研究所)
* **黃柏瑜** (國立中興大學 資訊管理研究所)

---

## 📌 專案簡介 (Project Overview)
面對總經環境劇變與市場的高波動性（如 2026 年 3 月的極端市場壓力與 VIX 滯留警戒區間），傳統的靜態資產配置（如 60/40 股債配置）與落後的計量指標已無法有效應對非線性的 **Regime Shift（市場結構轉變）**，往往導致投資組合面臨毀滅性的最大回撤（Max Drawdown）。

本專案捨棄傳統計量經濟學，提出一個具備自我進化能力的 **端到端深度強化學習（End-to-End DRL）** 交易代理人。結合 LSTM 的時序萃取能力與 PPO 的連續控制優勢，打造能在股災發生前進行前瞻性風險降權的動態資產配置系統。

## 🧠 核心技術架構 (Proposed Architecture)
本系統採用 **LSTM + PPO 雙引擎架構**：

1. **The Forecaster (LSTM)**：
   * 負責萃取時間序列特徵（OHLCV）。
   * 敏銳捕捉 VIX 波動率與殖利率等總經指標，提前感知 Regime Shift 訊號。
2. **The Risk Controller (PPO)**：
   * 負責連續動作空間的部位權重分配（Action Space: -1 到 1）。
   * 取代單純預測模型，PPO 直接與環境互動並規避滑價（Slippage）與手續費損耗。

### 🛡️ 風險感知獎勵函數 (Risk-Aware Reward Function)
為符合實務法人機構對下行風險的嚴格控管，我們設計了帶有最大回撤懲罰的獎勵機制：
$$R_t = \text{Sharpe}_t - \alpha \times \text{Drawdown Penalty}_t$$
* **$\text{Sharpe}_t$**：動態夏普比率，追求經風險調整後的最大化報酬。
* **$\alpha$**：風險偏好調節閥（高 $\alpha$ 代表高度防禦，嚴控回撤幅度）。

---

## 📊 實驗規劃與驗證 (Experimental Setup)

### 數據區間 (Data Horizon)
採用滾動式驗證（Walk-Forward Validation）避免未來數據洩漏：
* **訓練集 (Training Set) 70%**：2015 - 2024（涵蓋多頭與震盪循環）
* **樣本外測試集 (OOS Test Set) 30%**：2025 - 2026（重點壓力測試：2026/03/04 台股暴跌極端區間）

### 基準模型對照 (Baseline Models)
本專案將與以下基準模型進行嚴格的績效對比（Ablation Study）：
1. **Buy & Hold**：被動持有大盤指數（0050 或 SPY）。
2. **Static 60/40 Portfolio**：傳統 60% 股票 / 40% 債券靜態配置。
3. **Pure LSTM Prediction**：拔除 RL 決策引擎的純預測模型（驗證 PPO 控制部位的必要性）。

### 預期結果 (Expected Findings)
* **下行保護**：在 2026 年極端壓力測試下，將 Max Drawdown 嚴格壓縮至 **< 15%**。
* **防禦性轉型**：Agent 成功學會在市場崩潰「前」，自主調降高風險資產權重。

---

## 📂 專案架構 (Repository Structure)
*(註：以下為預設架構，可依實際程式碼上傳後微調)*
```text
├── data/                  # 歷史 OHLCV 與總經指標數據
├── envs/                  # 自定義 Gymnasium 交易環境 (包含手續費與滑價邏輯)
├── models/                # LSTM 特徵提取器與 PPO 網路架構
├── notebooks/             # 數據探索 (EDA) 與回測視覺化 Jupyter Notebooks
├── train.py               # 模型訓練腳本
├── evaluate.py            # 樣本外壓力測試與績效計算腳本
├── requirements.txt       # 環境依賴套件
└── README.md              # 專案說明文件