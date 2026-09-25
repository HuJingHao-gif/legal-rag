# -*- coding: utf-8 -*-
"""
第2步，砍掉文件开头的目录。
目录只列"编/章/节"，从不列"第X条"，所以直接拿第一条条文标题当分界，之前的全丢。
坑：正文开头的"第一编 总则""第一章 基本规定"和目录条目一个样子，被一起砍了，
    结果前 204 条缺编名、前 12 条缺章名，只能后面用 chunk.py 的 PATCH 硬补。
输入  data/民法典_去页家具.txt
输出  data/民法典_去目录.txt
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent
src = DATA_DIR / "民法典_去页家具.txt"
dst = DATA_DIR / "民法典_去目录.txt"

# 整行就是"第X条"，X 是汉字数字，不带标点
ART_TITLE = re.compile(r"^第[一二三四五六七八九十百千零]+条$")

with open(src, encoding="utf-8") as f:
    lines = f.readlines()                     # 每行一个元素，含行尾 \n

# 找正文起点 = 第一个条文标题的下标
body_start = None
for i, line in enumerate(lines):
    if ART_TITLE.match(line.strip()):         # 先剥掉行尾 \n 再匹配
        body_start = i                        # 第一个就够，记下标直接 break
        break

if body_start is None:
    raise SystemExit("没找到条文标题，检查正则或输入文件")

body = lines[body_start:]                     # 从这里截到尾，前面目录全不要

with open(dst, "w", encoding="utf-8") as f:
    f.writelines(body)

print(f"输入的目录区+正文共 {len(lines)} 行")
print(f"正文起点在第 {body_start+1} 行（下标 {body_start}）→ 砍掉 {body_start} 行目录")
print(f"保留 {len(body)} 行，已存: {dst}")
print("--- 新文件开头 5 行 ---")
for l in body[:5]:
    print(l, end="")                          # 行自带 \n，end="" 免得再空一行
