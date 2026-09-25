import requests
import re
import time
from pathlib import Path

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Referer": "https://flkofd.npc.gov.cn/reader",
    "Origin": "https://flkofd.npc.gov.cn",
}

base_url = "https://flkofd.npc.gov.cn/reader/text?file=http%253A%252F%252F172.16.220.27%253A38080%252Flaw-search%252FamazonFile%252FofdGenerateLink%253FfilePath%253Dprod%252F20200528%252F827f65fcb68f40cb941eed996c5212b0.ofd&_wr_timestamp=1788266660387&_wr_app_id=2396972e52c766e99770629cabe45e74&_wr_sign=ca35638269a9caf2068cf714bff761d868cfad70&_b=3.2.0&&_v=1&_i=1&_=1788266360885"

# 断点续传：断了就改这里接着跑（上次停在第 61 页就填 61）
# 页码从 0 开始！第 0 页是标题页（民法典这页 = 标题 + 目录前段）。
# 2026-09-15 才发现这里以前写死成 1，第 0 页一直漏抓。现在新抓取统一走 src/flk.py。
START_PAGE = 0

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
out = DATA_DIR / "民法典_raw.txt"

def page_to_text(d):
    """一页的字符数据拼成文本：area → line → chars"""
    lines = []
    for area in d.get("areas", []):
        for line in area.get("lines", []):
            lines.append("".join(c["char"] for c in line.get("chars", [])))
    return "\n".join(lines)

def fetch(page):
    """抓一页；超时自动重试，最多 3 次，全失败返回 None"""
    url = re.sub(r"_i=\d+", f"_i={page}", base_url)
    for attempt in range(1, 4):                    # 第 1、2、3 次
        try:
            return requests.get(url, headers=headers, timeout=30)   # 30 秒不回应算超时
        except requests.exceptions.RequestException:
            print(f"  第{page}页超时（第 {attempt}/3 次），3 秒后重试...")
            time.sleep(3)
    return None

# "a" 追加：不清空已有内容，新页接着往后写
with open(out, "a", encoding="utf-8") as f:
    page = START_PAGE
    while True:
        resp = fetch(page)
        if resp is None:
            print(f"第{page}页重试 3 次仍失败。把 START_PAGE 改成 {page} 再跑即可继续。")
            break
        if resp.status_code != 200:
            print(f"第{page}页 HTTP {resp.status_code}，结束")
            break
        data = resp.json()
        if data.get("index") != page or not data.get("areas"):
            print(f"第{page}页没有内容，结束")
            break
        text = page_to_text(data)
        f.write(f"\n===== 第{page}页 =====\n{text}")   # 边抓边存
        f.flush()                                      # 立刻落盘，崩了也不丢
        print(f"第{page}页：{len(text)} 字")
        page += 1
        time.sleep(0.5)

print("已存文件：", out)
