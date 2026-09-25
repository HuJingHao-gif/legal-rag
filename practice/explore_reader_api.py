import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Referer": "https://flkofd.npc.gov.cn/reader",
    "Origin": "https://flkofd.npc.gov.cn",
}

url = "https://flkofd.npc.gov.cn/reader/text?file=http%253A%252F%252F172.16.220.27%253A38080%252Flaw-search%252FamazonFile%252FofdGenerateLink%253FfilePath%253Dprod%252F20200528%252F827f65fcb68f40cb941eed996c5212b0.ofd&_wr_timestamp=1788266660387&_wr_app_id=2396972e52c766e99770629cabe45e74&_wr_sign=ca35638269a9caf2068cf714bff761d868cfad70&_b=3.2.0&&_v=1&_i=1&_=1788266360885粘贴你复制的完整URL"   # reader/text 的整条 URL 粘这里

# 先请求一页，看通不通
r = requests.get(url, headers=headers)
print("① 状态码：", r.status_code)
d = r.json()
print("   这一页 index =", d.get("index"), "| id =", d.get("id"))

# 抠出当前 _i，改成 +1，验证翻页
m = re.search(r"_i=(\d+)", url)      # 从 URL 里找 _i=数字
cur = int(m.group(1))                # 当前页码
next_url = url.replace(f"_i={cur}", f"_i={cur+1}")   # 页码 +1

r2 = requests.get(next_url, headers=headers)
d2 = r2.json()
print(f"② 改成 _i={cur+1} 后：")
print("   状态码：", r2.status_code, "| index =", d2.get("index"), "| id =", d2.get("id"))
