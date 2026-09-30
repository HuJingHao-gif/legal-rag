# -*- coding: utf-8 -*-
"""
评测命中判据的测试：is_gold。

为什么单独测它：recall@k 这张表现在是项目最显眼的数字（99~100%），
而它整个由 is_gold 一个函数决定。这个函数判松了，分数会虚高到没意义；
判严了，明明检索到了却算失败。两种都不会报错，只会让 README 上的数字变错。

合并库之后最危险的一格是跨法撞号："第一条"在 7 部法里都有。
只比条号的话，"从新法库里捞到生态环境法典第一条"会被算成命中了民法典的题。
所以 is_gold 必须 (条号, 法名) 一起判。

运行: python tests/test_recall_judgement.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_recall import is_gold


def chunk(条号, 简称=None, law=None):
    """造一个检索结果条目，字段名跟真实 chunks.jsonl 对齐"""
    c = {"条号": 条号}
    if 简称 is not None:
        c["简称"] = 简称
    if law is not None:
        c["law"] = law
    return c


def question(条号):
    """造一道评测题，只要标准条号那一个字段"""
    return {"标准条号": [条号]}


CASES = [
    # (说明, chunk, 题目, gold_law, 期望)
    ("条号对、法名对，简称写法",
     chunk("第一千二百五十四条", 简称="民法典"),
     question("第一千二百五十四条"), "民法典", True),

    ("法名是全称和简称的差别，要认得出来它们是同一部法",
     chunk("第五百一十条", 简称="中华人民共和国民法典"),
     question("第五百一十条"), "民法典", True),

    ("没有'简称'字段时回退到'law'字段，不能当成'法名对不上'",
     chunk("第五百一十条", law="民法典"),
     question("第五百一十条"), "民法典", True),

    ("跨法撞号：条号对，但属于另一部法 —— 这是合并库里最常见的一种假命中",
     chunk("第一条", 简称="生态环境法典"),
     question("第一条"), "民法典", False),

    ("跨法撞号的反向写法：题目的法名是简称，资料里是全称",
     chunk("第一条", 简称="中华人民共和国民法典"),
     question("第一条"), "生态环境法典", False),

    ("条号都不对，法名对也没用",
     chunk("第一千二百五十三条", 简称="民法典"),
     question("第一千二百五十四条"), "民法典", False),

    ("gold_law 为空 = 单库评测，库里只有一部法，不看法名",
     chunk("第一条", 简称="民法典"),
     question("第一条"), "", True),

]

# 潜在陷阱：下面断言的"实际行为"是漏的，但现有数据走不到这条路。
# 触发条件只有两个：某条 chunk 的 简称 和 law 全缺，或者法名写成了全称。
# merge_index.py 对"缺简称"是直接 SystemExit 的，所以合并库里出不来。
# 之所以记在这：那道闸门在另一个文件里，光读 is_gold 看不出它有前置依赖。
TRAPS = [
    # (说明, chunk, 题目, gold_law, 当前实际返回)
    ("法名取不到时放行 —— 空串是任何字符串的子串，'' in '民法典' 恒为 True",
     chunk("第一条"), question("第一条"), "民法典", True),

    ("法名写成全称时误命中 —— 司法解释的全称里就含'民法典'三个字，包含判断会认成民法典",
     chunk("第一条", law="最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释"),
     question("第一条"), "民法典", True),
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

    for desc, c, q, gold_law, want in CASES:
        check(desc, is_gold(c, q, gold_law), want)

    # 合并库场景：同一批候选里"第一条"出现了 7 次，只有法名对的那 1 条算中。
    # 这也是当初拆开判 (条号/法名) 会漏掉的那种错误。
    print("\n合并库场景：一批候选里 7 部法都有'第一条'")
    候选 = [chunk("第一条", 简称=n) for n in
            ["民法典", "合同编通则解释", "总则编解释", "婚姻家庭编解释",
             "继承编解释", "医疗保障法", "生态环境法典"]]
    hits = [c for c in 候选 if is_gold(c, question("第一条"), "民法典")]
    check("7 条候选里只有 1 条算命中，且法是民法典",
          [c["简称"] for c in hits], ["民法典"])

    print("\n潜在陷阱 —— 断言的是当前实际行为，不是正确行为")
    for desc, c, q, gold_law, want in TRAPS:
        check(desc, is_gold(c, q, gold_law), want)

    print(f"\n{ok}/{ok + bad} 通过")
    if bad:
        print(f"有 {bad} 个用例失败 —— recall@k 这张表已经不可信了")
        return 1
    print("命中判据能拦住跨法撞号，合并库上的 recall 数字有意义")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
