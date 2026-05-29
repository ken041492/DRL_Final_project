import requests

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoia2VuMDQxNDkyIiwiZW1haWwiOiJhbmR5MDQxNDkyQGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjB9.n1vtlxQBm1wtb_M-T33kTkQcGksd7gtfzWk9TuyQXrs"

headers = {"Authorization": f"Bearer {token}"}

url = "https://api.web.finmindtrade.com/v2/user_info"

payload = {

    "token": token,

}

resp = requests.get(url, headers=headers)

print(resp.json()["user_count"])  # 使用次數

print(resp.json()["api_request_limit"])  # api 使用上限