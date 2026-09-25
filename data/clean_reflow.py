# -*- coding: utf-8 -*-
"""
第3步，把断行拼回"一条一段"。改了好几版，这版是普查去目录的产物、又追了附则的 bug 之后定的。
OFD 是按行渲染的，正文被切得很碎，中文没有断词空格，拼的时候直接 join、不加空格。

三条规则：
  1. "第X条" 标题行，加上往下直到 下一条/结构标题/竖排单字 的正文行，拼成一段
  2. "第X编/章/节/附则" 加名字，拼成一行，名字可能横排一行，也可能是竖排的单字
  3. 竖排单字单独成行（附/则、监/护）= 标题字，绝不能粘进上一条正文

坑：竖排的篇章标题被按列切成单字行，比如"附则"变成"附"和"则"各占一行。
    早期没拦，这些标题字被当正文拼进了前一条，读起来还挺顺，就是平白多几个字，很难发现。
    碰到单字行一律当标题断点，不许往下拼。

输出每条两行：
  第一条
  为了保护……根据宪法，制定本法。
输入  data/民法典_去目录.txt
输出  data/民法典_按条.txt
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent
src = DATA_DIR / "民法典_去目录.txt"
dst = DATA_DIR / "民法典_按条.txt"

# 三种行各一条正则
# 必须整行匹配。早期图省事只判"这行含'条'字"，数出来 1062 条，真实是 1260
ART = re.compile(r"^第[一二三四五六七八九十百千零]+条$")              # 条文标题
DIV = re.compile(r"^(?:第[一二三四五六七八九十百千零]+(?:编|分编|章|节)|附则)$")  # 带数字头的结构标题 / 整块"附则"
ONE = re.compile(r"^[一-鿿]$")                             # 单个汉字（竖排标题字）

with open(src, encoding="utf-8") as f:
    lines = [l.strip() for l in f.readlines()]

out = []
i = 0
n = len(lines)
while i < n:
    s = lines[i]
    if not s:
        i += 1
        continue

    if ART.match(s):
        # 条文标题：往下收正文，碰到下一条 / 结构标题 / 竖排单字就停
        title = s
        frags = []
        i += 1
        while i < n:
            t = lines[i]
            if not t:
                i += 1
                continue
            if ART.match(t) or DIV.match(t) or ONE.match(t):
                break                 # 下一条 / 新标题 / 竖排字，本条到此为止
            frags.append(t)
            i += 1
        out.append(title)
        out.append("".join(frags))    # 断行拼接：中文没空格，直接连
        continue

    if DIV.match(s):
        # 结构标题：往下收名字，碰到下一条 / 结构标题就停
        head = s
        i += 1
        name_frags = []
        while i < n:
            t = lines[i]
            if not t:
                i += 1
                continue
            if ART.match(t) or DIV.match(t):
                break
            name_frags.append(t)      # 名字可能是"自然人"，也可能是竖排的"监""护"
            i += 1
        out.append(head + ("　" + "".join(name_frags) if name_frags else ""))
        continue

    if ONE.match(s):
        # 竖排标题字孤零零梗在正文里（附/则）：把连续的单字并回一行
        chars = []
        while i < n and ONE.match(lines[i]):
            chars.append(lines[i].strip())
            i += 1
        out.append("".join(chars))
        continue

    # 兜底，正常到不了：非标题行都该在上面被吃掉
    out.append(s)
    i += 1

with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(out))

arts = [o for o in out if ART.match(o)]
print(f"输出总行数: {len(out)}")
print(f"条文标题数: {len(arts)}   (应为 1260)")
print(f"已存: {dst}")
