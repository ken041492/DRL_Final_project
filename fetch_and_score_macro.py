import yaml
import time
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from FinMind.data import DataLoader

load_dotenv()
from tqdm import tqdm

# 初始化 tqdm 以便與 pandas 搭配使用
tqdm.pandas(desc="LLM 評估進度")

# ─── 🎯 財經新聞白名單 (主要報導即時盤勢與總經，過濾掉專欄與教學雜訊) ───
VALID_SOURCES = ['經濟日報', '工商時報', 'Anue鉅亨', '鉅亨網財經新聞', 'Yahoo奇摩股市', '中央社']

def get_sentiment_score(title):
    """
    任務二：設計大盤專用的 LLM 打分函數 (加入嚴格總經與雜訊過濾規則)
    """
    prompt = f"""你是一位總體經濟與量化基金分析師。請根據以下新聞標題，判斷其對『台灣整體股市大盤 (TAIEX)』的短期多空影響。
【嚴格過濾規則】：
1. 若標題為「投資教學、存股心得、ETF比較、名家專欄、事後覆盤（如：0050是什麼、教你怎麼存、為何大漲）」，請一律給予 0.0 分（中立）。
2. 只有在標題包含「具體的突發事件、宏觀數據公布、外資買賣超、央行決策」時，才給予非 0 分數。
分數範圍為 -1.0 (極度利空) 到 1.0 (極度利多)。
你只能回傳一個 JSON 格式的結果，格式為 {{"score": 分數}}。

新聞標題：{title}"""

    # 🛑 保持原設定，完全不動 payload 區塊
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 15,
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        result_str = data.get("response", "{}")
        
        result_json = json.loads(result_str)
        score = float(result_json.get("score", 0.0))
    except Exception as e:
        print(f"\n[錯誤] LLM 評估發生錯誤 ({title[:10]}...): {e}")
        score = 0.0
        
    # 每次打分完 sleep 0.3 秒，保護本地端 API 穩定度
    time.sleep(0.3)
    
    return score

def main():
    print("讀取設定檔...")
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    CURRENT_USER = config.get('settings', {}).get('current_user', '')
    STOCK_ID = config.get('settings', {}).get('stock_id', '0050')
    START_DATE = config.get('settings', {}).get('start_date', '2020-01-01')
    END_DATE = config.get('settings', {}).get('end_date', '2026-05-26')
    
    token = os.getenv(f"FINMIND_TOKEN_{CURRENT_USER.upper()}")
    
    if not token:
        print(f"找不到使用者 {CURRENT_USER} 的 Token，請確認 .env 設定 (FINMIND_TOKEN_{CURRENT_USER.upper()})。")
        return
        
    print(f"初始化 FinMind DataLoader (User: {CURRENT_USER}, Stock: {STOCK_ID})")
    dl = DataLoader()
    dl.login_by_token(api_token=token)
    
    # 建立斷點續傳機制
    processed_file = f"processed_dates_{STOCK_ID}.txt"
    processed_dates = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            for line in f:
                date_str = line.strip()
                if date_str:
                    processed_dates.add(date_str)
    
    csv_filename = f"{STOCK_ID}_macro_news_with_scores_{START_DATE}_to_{END_DATE}.csv"
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    delta = end_dt - start_dt
    
    print(f"開始爬取 {START_DATE} 到 {END_DATE} 的新聞並進行 LLM 評估...")
    
    for i in range(delta.days + 1):
        current_dt = start_dt + timedelta(days=i)
        current_date_str = current_dt.strftime("%Y-%m-%d")
        
        if current_date_str in processed_dates:
            print(f"[{current_date_str}] 已處理過，略過。")
            continue
            
        print(f"[{current_date_str}] 正在爬取新聞...")
        try:
            news_df = dl.taiwan_stock_news(stock_id=STOCK_ID, start_date=current_date_str, end_date=current_date_str)
            
            # 速率限制防護
            if isinstance(news_df, dict) and "limit" in str(news_df).lower():
                print(f"\n⚠️ 觸發 FinMind API 速率限制 (回傳含 limit)，安全中斷迴圈。")
                break
                
            if news_df is not None and not news_df.empty:
                # ─── 🛠️ 核心修改：先過濾新聞來源白名單 ───
                news_df = news_df[news_df['source'].isin(VALID_SOURCES)].copy()
                
                # 檢查過濾白名單後，當天是否還有剩餘新聞
                if not news_df.empty:
                    print(f"[{current_date_str}] 取得 {len(news_df)} 則即時財經新聞，開始進行 LLM 評估...")
                    
                    # 呼叫更新過 prompt 的打分函數
                    news_df['sentiment_score'] = news_df['title'].progress_apply(get_sentiment_score)
                    
                    # 追加寫入 CSV
                    write_header = not os.path.exists(csv_filename)
                    news_df.to_csv(csv_filename, mode='a', index=False, encoding="utf-8-sig", header=write_header)
                    print(f"[{current_date_str}] 當日新聞評估完成並已寫入 {csv_filename}。")
                else:
                    print(f"[{current_date_str}] 當天雖有新聞，但均非白名單來源，已全數過濾。")
            else:
                print(f"[{current_date_str}] 當天無新聞。")
                
            # 成功處理（或無有效新聞）後，記錄進度
            with open(processed_file, "a", encoding="utf-8") as f:
                f.write(current_date_str + "\n")
            processed_dates.add(current_date_str)
            
            time.sleep(0.3)
            
        except Exception as e:
            err_msg = str(e).lower()
            if "limit" in err_msg:
                print(f"\n⚠️ 觸發 FinMind API 速率限制 (Exception 含 limit)，安全中斷迴圈。錯誤細節：{e}")
                break
            else:
                print(f"[{current_date_str}] 發生未預期錯誤: {e}")
                time.sleep(1.0)

    print("\n✅ 所有指定日期的爬取與 LLM 評分流程已結束。")

if __name__ == "__main__":
    main()