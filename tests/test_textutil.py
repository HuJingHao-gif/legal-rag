# -*- coding: utf-8 -*-
"""
条号判据的测试：中文数字换算 + 条号提取。

为什么先测这个：cn2int 是**引用侧**的地基判据。
模型可能写"第1254条"也可能写"第一千二百五十四条"，要比对就得先换算成整数。
它算错一个字，引用校验和生成评测会一起错，而且不会抛异常 —— 只是数字悄悄不对。

注意别把它当成整条链的地基：检索评测的 is_gold() 走的是**原始字符串相等**
（c["条号"] != q["标准条号"][0]），根本不经过 cn2int，它假设评测集和语料
两边的条号写法完全一致。所以改这里不会影响 recall@k。

运行: python tests/test_textutil.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from textutil import cn2int, articles_in, ART_RE


CN2INT_CASES = [
    # (说明, 输入, 期望)
    ("一位数",                 "第一条",           1),
    ("整十，'十'开头要算 10 而不是 1", "第十条",     10),
    ("二十几",                 "第二十条",         20),
    ("九十几次，不进位的百位边界", "第九十九条",      99),
    ("整百",                   "第二百条",         200),
    ("带'零'的三位数",          "第一百零八条",      108),
    ("跨千带'零'",             "第一千零一条",      1001),
    ("四位数（民法典典型条号）",  "第一千二百五十四条", 1254),
    ("民法典最后一条",          "第一千二百六十条",   1260),
    ("阿拉伯数字写法，要跟中文写法等价", "第1254条", 1254),
    ("纯数字字符串",            "1254",           1254),
    ("整数入参（条号字段有时就是 int）", 1254,        1254),
]

# 这里每条都是"一段文本 → 应该抠出哪些条号"
ARTICLES_CASES = [
    ("一行里两个引用", "依据《民法典》第一百四十三条、第五百一十条处理", {143, 510}),
    ("中文与阿拉伯数字混排", "见第1254条，另见《民法典》第一条", {1254, 1}),
    ("空串", "", set()),
    ("None（正文缺失时的兜底，不能崩）", None, set()),
    ("没有条号的普通句子", "该行为不构成侵权", set()),
]

# 「第X条之一」—— 刑法和修正案的常见写法，它和「第X条」是两条不同的条文。
# 09-29 修：以前是两个静默错值，现在一律抛错停住。
#   cn2int("第十条之一") 曾经返回 11（不认识的字符被跳过，"之一"的"一"被当成数字累加）
#   articles_in / ART_RE 曾经把 "第十条之一" 当成 "第十条"（正则只截到"条"）
# 为什么不顺手把它支持上：得改 cn2int 的整数编码（比如 条号*1000+支号），
# 那会动到召回分桶、条号比大小等已有口径，等于要连评测一起重做。
# 现有 7 部法都没有这种写法，所以先让它大声失败；等真要加刑法语料时再一起实现。
#
# 注意 ART_RE 单独用仍然只认「第X条」（下面最后一条用例锁着这个事实），
# 所以拦截必须发生在调用点（cn2int / articles_in），光靠 ART_RE 拦不住。
RAISE_CASES = [
    # (说明, 会抛错的调用, 报错消息里必须出现的关键词)
    ("cn2int 直接吃 '第十条之一' → 抛错，不再静默算成 11",
     lambda: cn2int("第十条之一"), "条之一"),
    ("articles_in 遇到 '第十条之一' → 抛错，不再静默当成 '第十条'",
     lambda: articles_in("《刑法》第十条之一"), "条之一"),
    ("cn2int 遇到别的多余字（如 '第十条第二款'）→ 抛错，不再静默跳过",
     lambda: cn2int("第十条第二款"), "不认识的字符"),
]


def main():
    ok = bad = 0

    def check(desc, got, want):
        nonlocal ok, bad
        if got == want:
            ok += 1
            print(f"  ✓ {desc}")
        else:
            bad += 1
            print(f"  ✗ {desc}")
            print(f"      期望 {want}，实际 {got}")

    print("cn2int —— 中文数字 / 阿拉伯数字 → 整数")
    for desc, src, want in CN2INT_CASES:
        check(f"{desc}（{src} → {want}）", cn2int(src), want)

    print("\narticles_in —— 从一段文本里抠出所有条号")
    for desc, src, want in ARTICLES_CASES:
        check(desc, articles_in(src), want)

    print("\n「第X条之一」—— 必须大声失败，不许静默算成「第X条」")

    def check_raises(desc, call, keyword):
        nonlocal ok, bad
        try:
            got = call()
        except ValueError as e:
            if keyword in str(e):
                ok += 1
                print(f"  ✓ {desc}")
            else:
                bad += 1
                print(f"  ✗ {desc}\n      抛了 ValueError，但消息里没有「{keyword}」：{e}")
        except Exception as e:
            bad += 1
            print(f"  ✗ {desc}\n      期望 ValueError，实际抛 {type(e).__name__}：{e}")
        else:
            bad += 1
            print(f"  ✗ {desc}\n      期望抛 ValueError，实际没抛、返回了 {got!r}")

    for desc, call, keyword in RAISE_CASES:
        check_raises(desc, call, keyword)

    # 这条不是在测正确行为，而是在锁"拦截必须放在调用点"这个前提：
    # ART_RE 自己会静默截断，谁绕过 cn2int/articles_in 直接用正则，谁就绕过了拦截。
    check("ART_RE 单独用仍只认「第X条」（截断事实，不是待修的 bug）",
          ART_RE.findall("《刑法》第十条之一"), ["第十条"])

    print(f"\n{ok}/{ok + bad} 通过")
    if bad:
        print(f"有 {bad} 个用例失败 —— 条号判据出了问题，引用校验和评测都会跟着错")
        return 1
    print("条号判据在现有语料的条号范围内是可靠的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
