import os
import time
import yaml
import pandas as pd
from FinMind.data import DataLoader

# ─── 📦 讀取 config.yaml 設定檔 ───
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ─── 🔑 提取設定值 ───
CURRENT_USER = config["settings"]["current_user"]
STOCK_ID = config["settings"]["stock_id"]
START_DATE = config["settings"]["start_date"]
END_DATE = config["settings"]["end_date"]

# 動態從 Token 清單中抓取當前用戶的 Token
TOKEN = config["finmind_tokens"].get(CURRENT_USER)

if not TOKEN:
    raise ValueError(f"❌ 在 config.yaml 的 finmind_tokens 中找不到用戶 [{CURRENT_USER}] 的 Token！")

print(f"🔑 成功讀取設定！當前用戶: [{CURRENT_USER}]，準備撈取標的: {STOCK_ID}")

dl = DataLoader(token=TOKEN)

OUTPUT_FILE = f"{STOCK_ID}_history_news_2025_2026.csv"
PROGRESS_FILE = f"processed_dates_{STOCK_ID}.txt"  # 📝 新增一個專門記進度的筆記本

# 1. 定義總時間範圍
START_DATE = "2025-06-26"
END_DATE = "2026-05-15"
full_date_range = pd.date_range(start=START_DATE, end=END_DATE).strftime("%Y-%m-%d")

# 2. 檢查進度筆記本，載入所有「已經查過」的日期
processed_dates = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        processed_dates = set(line.strip() for line in f if line.strip())
    print(f"ℹ️ 偵測到進度紀錄，已完成 {len(processed_dates)} 天的歷史排查，將自動跳過。")

print(f"🚀 開始分批下載任務...")

# 3. 跑迴圈，只抓「完全沒查過」的日期
for idx, current_date in enumerate(full_date_range):
    if current_date in processed_dates:
        continue  # 查過了就無情跳過（不論當初有沒有新聞）

    try:
        df_single_day = dl.taiwan_stock_news(stock_id=STOCK_ID, start_date=current_date)
        
        # A. 如果這天有新聞，立刻追加寫入大 CSV 檔案
        if not df_single_day.empty:
            file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
            df_single_day.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
            print(f"[{idx+1}/{len(full_date_range)}] {current_date} 抓取成功並寫入！拿到 {len(df_single_day)} 筆。")
        else:
            print(f"[{idx+1}/{len(full_date_range)}] {current_date} 當天無新聞。")
            
        # 💡 核心優化：只要 API 成功回應（不論有沒有新聞），就立刻把這天寫進進度筆記本
        with open(PROGRESS_FILE, "a") as f:
            f.write(f"{current_date}\n")
            
    except Exception as e:
        print(f"❌ {current_date} 發生錯誤: {e}")
        if "reach the upper limit" in str(e).lower() or "limit" in str(e).lower():
            print("\n⚠️ 觸發 FinMind 每小時 600 次限制了！程式安全中斷。")
            print("💡 進度已安全儲存！請等待 30 ~ 60 分鐘後再次執行，會精準從這一開續傳！")
            break

    # 稍微拉長冷卻時間至 0.2 秒，走得慢反而能走得遠
    time.sleep(0.2)

print("\n🏁 目前階段任務執行完畢。")