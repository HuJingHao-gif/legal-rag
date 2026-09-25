# -*- coding: utf-8 -*-
"""
分块：把 民法典_按条.txt（每条两行）切成 chunk。
chunk 单位就取"一条"——法律规范本身自足，粒度正好对上"引用法条"。
每条记 法律名 / 编 / 分编 / 章 / 节 / 条号 / 正文 / 检索文本，写 JSONL，一行一个。

状态机：一路往下读，
  读到"第X编/分编/章/节"标题，就更新当前层级上下文（编换了，下层的分编/章/节要清空）
  读到"第X条"加下一行正文，就出一条 chunk，把当前上下文一起带上
输入  data/民法典_按条.txt
输出  data/民法典_chunks.jsonl
"""
import re
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
src = DATA_DIR / "民法典_按条.txt"
dst = DATA_DIR / "民法典_chunks.jsonl"

LAW = "中华人民共和国民法典"

ART  = re.compile(r"^第[一二三四五六七八九十百千零]+条$")
HEAD = re.compile(r"^(第[一二三四五六七八九十百千零]+(?:编|分编|章|节)|附则)(?:　(.+))?$")

# 结构补丁
# 已知缺口：第2步砍目录时，正文开头的"第一编 总则""第一章 基本规定"和目录条目
#          一模一样，被一起砍了，前 204 条缺编名、前 12 条缺章名。
# 这里按已知事实把起始层级垫上，状态机跑到"第二章""第二编"会自己覆盖。
# 以后哪部法也有类似缺口，只改这个 dict，别动下面的算法。
PATCH = {
    "编": "第一编　总则",
    "章": "第一章　基本规定",
}

with open(src, encoding="utf-8") as f:
    lines = [l.strip() for l in f.readlines()]

# 当前层级上下文：读到标题就更新它，读到条文就把它一起塞进 chunk
# 初值 = 上面那个补丁（文件开头的层级）
ctx = {"编": PATCH["编"], "分编": "", "章": PATCH["章"], "节": ""}

chunks = []
i = 0
n = len(lines)
while i < n:
    s = lines[i]
    if not s:
        i += 1
        continue

    m = HEAD.match(s)
    if m:
        token = m.group(1)                 # 如 "第二编" / "附则"
        name  = (m.group(2) or "").strip()  # 如 "物权"
        full  = (token + "　" + name).strip() if name else token
        if token.endswith("分编"):
            ctx["分编"] = full; ctx["章"] = ""; ctx["节"] = ""
        elif token.endswith("编"):
            ctx["编"] = full; ctx["分编"] = ""; ctx["章"] = ""; ctx["节"] = ""
        elif token.endswith("章"):
            ctx["章"] = full; ctx["节"] = ""
        elif token.endswith("节"):
            ctx["节"] = full
        else:                              # 附则：没有编号的"编级"标题
            ctx["编"] = "附则"; ctx["分编"] = ""; ctx["章"] = ""; ctx["节"] = ""
        i += 1
        continue

    if ART.match(s):
        # 下一条该是正文行；要是它本身就是标题，说明这条正文是空的，防御一下
        body = ""
        consumed = 1
        if i + 1 < n:
            nxt = lines[i + 1]
            if nxt and not ART.match(nxt) and not HEAD.match(nxt):
                body = nxt
                consumed = 2
        # 检索文本 = 法律名 + 章节路径 + 条号 + 正文，喂给 embedding 的就是它
        path = " ".join(x for x in [ctx["编"], ctx["分编"], ctx["章"], ctx["节"]] if x)
        embed_text = f"《{LAW}》{path} {s} {body}".strip()

        chunks.append({
            "law":  LAW,
            "编":   ctx["编"],
            "分编": ctx["分编"],
            "章":   ctx["章"],
            "节":   ctx["节"],
            "条号": s,
            "正文": body,
            "检索文本": embed_text,
        })
        i += consumed
        continue

    i += 1        # 兜底：认不出的行跳过，正常不该出现

with open(dst, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")   # ensure_ascii=False：中文原样存

print(f"chunk 总数: {len(chunks)}   (应为 1260)")
print(f"已存: {dst}")
