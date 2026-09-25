# -*- coding: utf-8 -*-
"""条号相关的文本工具：中文数字换算、条号提取、引用校验。eval_answer / answer 共用。"""
import re

# "第X条"，X 可以是中文数字，也可以是阿拉伯数字
ART_RE = re.compile(r"第[一二三四五六七八九十百千零\d]+条")

_CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn2int(s):
    """'第一千二百五十四条' / '第1254条' / 1254 → 1254（统一成整数才好比大小）"""
    s = str(s).strip().lstrip("第").rstrip("条")
    if s.isdigit():
        return int(s)
    total = section = num = 0
    for ch in s:
        if ch in _CN:
            num = _CN[ch]
        elif ch in _UNIT:
            section += (num or 1) * _UNIT[ch]     # "十"开头 → 10
            num = 0
        elif ch == "万":
            total += (section + num) * 10000
            section = num = 0
    return total + section + num


def articles_in(text):
    """从一段文本里抠出所有条号，返回整数集合"""
    return {cn2int(m) for m in ART_RE.findall(text or "")}
