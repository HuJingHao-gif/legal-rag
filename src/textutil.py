


# -*- coding: utf-8 -*-
"""条号相关的文本工具：中文数字换算、条号提取、引用校验。eval_answer / answer 共用。"""
import re

# "第X条"，X 可以是中文数字，也可以是阿拉伯数字
ART_RE = re.compile(r"第[一二三四五六七八九十百千零\d]+条")

# "第X条之一" 是刑法和修正案的写法。"第十条之一" 和 "第十条" 是两条不同的条文，
# 但当前的条号编码只有一个整数，表达不了这层区别 —— 两个会被算成同一条。
# 这里不硬猜，直接拦住：静默把两条算成一条，比报错危险得多。
ART_BRANCH_RE = re.compile(r"第[一二三四五六七八九十百千零\d]+条之[一二三四五六七八九十]+")

_CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def _reject_branch(text):
    """碰到「第X条之一」就停住。

    要真正支持它，得改 cn2int 的整数编码（比如 条号*1000+支号），
    而那会动到召回分桶、条号比大小等已有口径，得连评测一起重做。
    现有 7 部法语料都不用这种写法，所以这里选择大声失败。
    """
    m = ART_BRANCH_RE.search(text or "")
    if m:
        raise ValueError(
            f"遇到「{m.group(0)}」：「条之一」与「{m.group(0).split('之')[0]}」是两条不同的条文，"
            f"当前条号编码会把它们算成同一条。要加刑法这类含「之一」的法，先改 cn2int 的编码。"
        )


def cn2int(s):
    """'第一千二百五十四条' / '第1254条' / 1254 → 1254（统一成整数才好比大小）

    不认识的字符一律抛错，不再静默跳过 —— 跳过会把 '第十条之一' 算成 11，
    这种错值会一路静默流进引用校验和 recall，最后表现成"分数虚高"，很难往回查。
    """
    _reject_branch(str(s))
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
        else:
            raise ValueError(f"cn2int 不认识的字符 {ch!r}（输入 {s!r}）：条号只支持「第X条」")
    return total + section + num


def articles_in(text):
    """从一段文本里抠出所有条号，返回整数集合"""
    # 先拦截，因为 ART_RE 只截到"条"，"第十条之一"会被静默当成"第十条"
    _reject_branch(text)
    return {cn2int(m) for m in ART_RE.findall(text or "")}
