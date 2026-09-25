# -*- coding: utf-8 -*-
"""
第1步，清页家具。
OFD 是按页给的，每页带着 "===== 第N页 =====" 和页脚 "－108－"，中间还夹空行，
先把这些扔掉再谈别的。
输入  data/民法典_raw.txt
输出  data/民法典_去页家具.txt
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent           # 脚本就放 data/ 下，parent 即 data
src = DATA_DIR / "民法典_raw.txt"
dst = DATA_DIR / "民法典_去页家具.txt"

# 三种垃圾行，各一条正则
PAGE_MARK = re.compile(r"^===== 第\d+页 =====$")   # ===== 第7页 =====
FOOTER    = re.compile(r"^－\d+－$")               # －7－ （全角横线夹数字）
# 空行不走正则，单独用 line.strip() == "" 判

def is_garbage(line: str) -> bool:
    """是垃圾行就 True"""
    if PAGE_MARK.match(line):
        return True
    if FOOTER.match(line):
        return True
    if line.strip() == "":        # 空行 / 纯空格行
        return True
    return False

# 读进来逐行过滤再写出去
with open(src, encoding="utf-8") as f:
    raw_lines = f.readlines()

kept = [line for line in raw_lines if not is_garbage(line)]   # 只留非垃圾行

with open(dst, "w", encoding="utf-8") as f:
    f.writelines(kept)

# 砍了多少，打出来看
print(f"原始行数:   {len(raw_lines)}")
print(f"过滤后行数: {len(kept)}")
print(f"已保存:     {dst}")
