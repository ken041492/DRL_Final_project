import pandas as pd
import requests
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_sentiment_score(title):
    url = "http://localhost:11434/api/generate"
    prompt = f"""你是一位專業的金融分析師。請根據以下台積電的新聞標題，給予一個情緒分數。
分數範圍為 -1.0 (極度利空) 到 1.0 (極度利多)。0.0 表示中立。
你只能回傳一個 JSON 格式的結果，格式為 {{"score": 分數}}，不要回傳任何其他說明文字。

新聞標題：{title}
"""
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 15,    # 👈 核心優化 2：限制最大生成字數，強迫 LLM 快點閉嘴
            "temperature": 0.0     # 👈 核心優化 3：降低隨機性，速度最快
        }
    }
    
    try:
        # 縮短 timeout 防止某幾筆死鎖卡住整體進度
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        response_text = result.get('response', '')
        
        data = json.loads(response_text)
        score = float(data.get("score", 0.0))
        return max(-1.0, min(1.0, score))
    except Exception:
        return 0.0

def main():
    input_file = "tsmc_history_news_2025_2026.csv"
    output_file = "tsmc_news_with_scores_evaluate.csv"
    
    TEST_MODE = False
    TEST_LIMIT = 10
    
    # 👈 核心優化 4：設定並行執行緒數量（可根據你電腦的 CPU/GPU 效能調整，一般 4~8 最佳）
    MAX_WORKERS = 4 

    if not os.path.exists(input_file):
        print(f"找不到輸入檔案: {input_file}")
        return

    df = pd.read_csv(input_file)
    
    # 初始化分數欄位
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0

    # 斷點續傳：找出哪些還沒被跑過（值為 0.0 且不是第一筆，或者透過已存在的輸出檔比對）
    start_index = 0
    if os.path.exists(output_file):
        print(f"發現現有的輸出檔案 {output_file}，嘗試進行斷點續傳...")
        try:
            existing_df = pd.read_csv(output_file)
            start_index = len(existing_df)
            df.loc[:start_index-1, 'sentiment_score'] = existing_df['sentiment_score'].values
            print(f"已從第 {start_index} 筆開始繼續處理。")
        except Exception as e:
            print(f"讀取現有檔案失敗: {e}，將從頭開始執行。")
    
    # 決定處理區間
    target_len = min(start_index + TEST_LIMIT, len(df)) if TEST_MODE else len(df)
    tasks_indices = list(range(start_index, target_len))
    
    print(f"總共 {len(df)} 筆資料，即將使用 {MAX_WORKERS} 個執行緒處理 {len(tasks_indices)} 筆資料...")

    # 每 100 筆批量存檔用的計數器
    save_counter = 0

    # 👈 核心優化 5：使用 ThreadPoolExecutor 進行多任務並行發送
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 建立任務對照表 {future: index}
        future_to_index = {
            executor.submit(get_sentiment_score, str(df.loc[idx, 'title'])): idx 
            for idx in tasks_indices
        }
        
        # 使用 tqdm 追蹤並行進度
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="並行打分進度"):
            idx = future_to_index[future]
            try:
                score = future.result()
            except Exception:
                score = 0.0
            
            df.loc[idx, 'sentiment_score'] = score
            save_counter += 1
            
            if TEST_MODE:
                print(f"\n[測試] 標題: {df.loc[idx, 'title']} -> 評分: {score}")

            # 每滿 100 筆做一次安全的斷點存檔
            if save_counter % 100 == 0:
                # 這裡直接儲存整份進度
                df.to_csv(output_file, index=False)

    # 最終完整儲存
    df.to_csv(output_file, index=False)        
    print(f"全部處理完成！已儲存至 {output_file}")

if __name__ == "__main__":
    main()