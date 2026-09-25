import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Referer": "https://flk.npc.gov.cn/search",
    "Origin": "https://flk.npc.gov.cn",
}

url = "https://flk.npc.gov.cn/law-search/search/list"
body = {
    "searchContent": "民法典",   # 搜什么
    "pageNum": 1,                # 第几页
    "pageSize": 5,               # 每页几条
    "searchType": 2,
    "searchRange": 1,
    "sxrq": [], "gbrq": [], "gbrqYear": [], "sxx": [],
    "flfgCodeId": [], "zdjgCodeId": [],
    "orderByParam": {"order": "-1", "sort": ""},
}

resp = requests.post(url, headers=headers, json=body)
print("HTTP 状态码：", resp.status_code)
data = resp.json()
print("业务状态码：", data["code"])
print("返回消息：", data["msg"])
rows = data["rows"]
print("找到", data.get("total"), "条")
print("第一条：", rows[0]["title"], "| bbbs:", rows[0]["bbbs"])
