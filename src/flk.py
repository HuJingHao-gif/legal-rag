# -*- coding: utf-8 -*-
"""
国家法律法规数据库（flk.npc.gov.cn）通用客户端。

抽成单独模块是因为多法扩展要反复走"搜索 → 取详情 → 抓正文"，
封装好之后加新法只换参数，抓取逻辑不用重写。

2026-09-14 实测的坑，比任何文档都可靠：
  1. 搜索请求体里**没有 keyword 字段**了。现在的形状是
     {searchRange, searchType, searchContent, sxrq, gbrq, sxx, gbrqYear,
      flfgCodeId, zdjgCodeId, page, size}
     searchType: 1=按标题搜, 2=按正文搜。用旧字段（keyword）会 500"系统异常"。
  2. 正文接口 flkofd.npc.gov.cn/reader/text 只认 file + _i 两个参数，
     浏览器上那串 _wr_sign / _wr_app_id 是页面自己加的，**不校验，可以不要**。
  3. file 参数里的 inner URL 要**双重 URL-encode**（% 变成 %25）。
  4. 接口偶尔 ReadTimeout，是正常抖动，重试就行。
"""
import re
import time
from urllib.parse import quote

import requests

BASE = "https://flk.npc.gov.cn"
READER = "https://flkofd.npc.gov.cn/reader/text"
OFD_GATEWAY = "http://172.16.220.27:38080/law-search/amazonFile/ofdGenerateLink"

LIST_BODY = {
    "searchRange": 1, "sxrq": [], "gbrq": [], "searchType": 1, "sxx": [],
    "gbrqYear": [], "flfgCodeId": [], "zdjgCodeId": [], "searchContent": "",
    "page": 1, "size": 20,
}

_session = requests.Session()
_session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"),
    "Referer": f"{BASE}/fl.html",
    "Origin": BASE,
    "Content-Type": "application/json",
})


def _get(url, tries=4, **kw):
    """带重试的 GET —— flk 时不时 ReadTimeout，抖动是常态"""
    for i in range(tries):
        try:
            return _session.get(url, timeout=60, **kw)
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
            time.sleep(4)


def _post(url, tries=4, **kw):
    for i in range(tries):
        try:
            return _session.post(url, timeout=60, **kw)
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
            time.sleep(4)


def search(keyword, search_type=1, page=1, size=20, **filters):
    """搜法规。search_type: 1=标题, 2=正文。返回 [{bbbs,title,flxz,gbrq,sxx}, ...]

    flxz 就是层级类型：宪法 / 法律 / 行政法规 / 地方性法规 / 司法解释 ——
    只保留"法律 + 行政法规 + 司法解释"就能滤掉 8000 多部地方性法规。
    """
    body = dict(LIST_BODY)
    body.update({"searchContent": keyword, "searchType": search_type, "page": page, "size": size})
    body.update(filters)
    data = _post(f"{BASE}/law-search/search/list", json=body).json()
    rows = data.get("rows") or []
    for r in rows:
        r["title"] = re.sub(r"<[^>]+>", "", r.get("title") or "")   # 去掉 <em class='highlight'>
    return rows


def detail(bbbs):
    """取元信息 + 目录树 + ossFile。**没有正文**，正文在 OFD 里。"""
    return _get(f"{BASE}/law-search/search/flfgDetails", params={"bbbs": bbbs}).json()["data"]


def ofd_reader_url(ofd_path, page=1):
    """把 ossWordOfdPath 拼成逐页正文接口的 URL"""
    inner = f"{OFD_GATEWAY}?filePath={ofd_path}"
    return f"{READER}?file={quote(quote(inner, safe=''), safe='')}&_i={page}"


def page_to_lines(data):
    """一页的 JSON → 文本行。结构是 area → line → chars，OFD 的原生单位就是"坐标上的字"。"""
    out = []
    for area in data.get("areas") or []:
        for line in area.get("lines") or []:
            out.append("".join(c["char"] for c in line.get("chars") or []))
    return out


def fetch_text(bbbs, sleep=0.5, max_pages=999, verbose=True, start_page=0):
    """按页抓全文，返回 (行列表, 用了多少页)。

    start_page 默认 0 —— 页码实测从 0 开始，而且**第 0 页是标题页**：
    民法典第 0 页 = 标题 + 目录前段，司法解释第 0 页 = 标题 + 通过日期 + 第一、二条。
    老脚本 data/fetch.py 写死 START_PAGE=1，所以漏了第 0 页。民法典侥幸没事
    （第 0 页只有标题和目录，目录反正要砍），但司法解释没目录，漏第 0 页 = 直接丢正文。
    """
    d = detail(bbbs)
    ofd = (d.get("ossFile") or {}).get("ossWordOfdPath")
    if not ofd:
        raise SystemExit(f"{bbbs} 没有 ossWordOfdPath，取不到正文")

    lines, page = [], start_page
    while page <= max_pages + start_page:
        try:
            resp = _get(ofd_reader_url(ofd, page))
        except requests.exceptions.RequestException:
            if verbose:
                print(f"  第{page}页重试耗尽，停在这里")
            break
        if resp.status_code != 200:
            break
        data = resp.json()
        if data.get("index") != page or not data.get("areas"):
            break                                  # 没有内容的页 = 到头了
        lines += page_to_lines(data)
        if verbose:
            print(f"  第{page}页 {len(data['areas'])} 区")
        page += 1
        time.sleep(sleep)                          # 对政府网站礼貌一点
    return lines, page - start_page


if __name__ == "__main__":
    for r in search("审理民间借贷案件适用法律若干问题的规定")[:5]:
        print(f'[{r.get("flxz")}] {r["title"][:50]} | {r.get("gbrq")} | {r["bbbs"]}')
