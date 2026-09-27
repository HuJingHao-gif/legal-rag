# -*- coding: utf-8 -*-
"""
引用校验的逻辑测试：喂已知的好/坏引用，看它判得对不对。

为什么单独测这个：端到端跑 200 题（多跳 + 新法）编造率是 0，但那**不是**在测校验器 ——
那测的是"系统整体不编"，起作用的是提示词里的"只能依据资料回答"。模型听话了，
校验器根本没被触发。

校验器是**提示词失效时的兜底**。要验它，必须主动构造坏引用喂进去。
它平时不响，不代表它不需要正确 —— 恰好相反，兜底的东西坏了最难发现。

运行: python tests/test_citation_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from answer import _validate


def hit(no, law="民法典", body="（条文正文，测试里不重要）"):
    """造一条假的"检索到的资料"。返回 (分数, chunk) —— 跟 search() 的输出同形。"""
    return (0.9, {"条号": no, "简称": law, "正文": body})


# 资料：只喂了三部法各一条
HITS = [
    hit("第一百四十三条"),
    hit("第五百一十条"),
    hit("第一条", law="生态环境法典"),
]


CASES = [
    # (说明, 模型给的引用, 期望通过?, 期望说明里包含什么)
    ("引用正确：条号在资料里、法名也对",
     ["《民法典》第一百四十三条"], True, None),

    ("引用正确：资料里的另一部法",
     ["《生态环境法典》第一条"], True, None),

    ("★ 编造：条号根本不在资料里",
     ["《民法典》第九百九十九条"], False, "不在检索到的资料里"),

    ("★ 跨法撞号：条号在资料里，但属于另一部法",
     ["《民法典》第一条"], False, "法律名对不上"),

    ("★ 跨法撞号的反向：资料里是生态环境法典第一条，模型说是民法典",
     ["《生态环境法典》第五百一十条"], False, "法律名对不上"),

    ("格式坏：整条引用解析不出条号",
     ["根据相关规定处理"], False, "无法解析"),

    ("混合：一条对一条错 —— 只要有一条错，整体就不通过",
     ["《民法典》第一百四十三条", "《民法典》第九百九十九条"], False, "不在检索到的资料里"),

    ("空引用：没做任何声明，不算错（无可核验的内容）",
     [], True, None),
]


def main():
    ok = bad = 0
    for desc, cites, want_pass, want_note in CASES:
        got_pass, notes = _validate(cites, HITS)
        problems = []
        if got_pass != want_pass:
            problems.append(f"期望{'通过' if want_pass else '不通过'}，实际{'通过' if got_pass else '不通过'}")
        if want_note and not any(want_note in n for n in notes):
            problems.append(f"说明里没提到「{want_note}」，实际：{notes}")

        if problems:
            bad += 1
            print(f"  ✗ {desc}")
            for p in problems:
                print(f"      {p}")
        else:
            ok += 1
            print(f"  ✓ {desc}")

    print(f"\n{ok}/{len(CASES)} 通过")
    if bad:
        print(f"★ 有 {bad} 个用例失败 —— 校验器的兜底逻辑有问题")
        return 1
    print("校验器对已知的坏引用都能识别")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
