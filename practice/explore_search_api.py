import requests
# 请求头：User-Agent / Referer 装成浏览器
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":"https://flk.npc.gov.cn/",
}

# 搜索是 POST，参数塞 body
url = "https://flk.npc.gov.cn/law-search/search/list"
body = {
    "keyword":"",
    "page": 1,
    "size": 5,
}

resp = requests.post(url,headers=headers,json = body)
print("状态码: ", resp.status_code)  # 200 才成功

data = resp.json()   # 解析成 dict
print(data)

"""
print("=" * 40)
print("状态码:", resp.status_code)     # 200 就说明成功了
print("=" * 40)
print("resp 身上挂着这些东西:")
print(dir(resp))
"""

