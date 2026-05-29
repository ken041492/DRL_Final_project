import pandas as pd
import glob

def main():
    # 抓取目錄下所有 0050_macro_news_with_scores 開頭的 CSV 檔
    file_list = glob.glob("0050_macro_news_with_scores_*.csv")
    print(f"找到 {len(file_list)} 個 CSV 檔案準備合併...")

    df_list = []
    for file in file_list:
        df = pd.read_csv(file)
        df_list.append(df)

    # 1. 垂直合併
    df_all = pd.concat(df_list, ignore_index=True)

    # 2. 轉換日期格式並排序 (確保時間序列的正確性)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')

    # 3. 去除重複值 (以 date 和 title 兩者皆相同作為重複標準，保留最後一筆)
    df_all = df_all.drop_duplicates(subset=['date', 'title'], keep='last')

    # 4. 輸出最終黃金檔案
    final_output = "macro_sentiment_2021_2026_final.csv"
    df_all.to_csv(final_output, index=False, encoding="utf-8-sig")
    
    print(f"✅ 合併清洗完成！")
    print(f"總共獲得 {len(df_all)} 筆有效的大盤總經新聞。")
    print(f"檔案已儲存為: {final_output}")

if __name__ == "__main__":
    main()