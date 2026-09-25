import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Referer": "https://flk.npc.gov.cn/detail?id=ff808081729d1efe01729d50b5c500bf",   # 骗一下 Referer
    "Origin": "https://flk.npc.gov.cn",
}

url = "https://flk.npc.gov.cn/prod/20200528/bd53dd912c1048f2aecbaa229238334b.pdf"

resp = requests.get(url, headers=headers)
print("状态码：", resp.status_code)
print("Content-Type：", resp.headers.get("Content-Type"))
print("内容大小：", len(resp.content), "字节")
print("开头 20 字节：", resp.content[:20])
