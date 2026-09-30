# -*- coding: utf-8 -*-
"""
评测集筛选判据的测试：has_leak（泄题）+ pick（分层配额）。

为什么单独测它：这两个函数决定"哪些题能进评测集"，而评测集一旦脏了，
后面所有数字都跟着脏，且没有任何一步会报错。

- has_leak 判松了 → 题目里直接带着答案 → recall 虚高
  （实测过反例：对"法条原话"类题查泄题，33 条全被误杀，所以它只对口语化题生效）
- pick 判错了 → 评测集偏科（2026-09-21 那次：新法 50 题里 48 题是医疗保障法）
  这类偏差不会让任何指标变红，只是让数字不再代表整个库

运行: python tests/test_filter_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from filter_eval import LEAK_N, has_leak, norm, pick


# 一段真实的法条正文，够长，能切出多个 12 字片段
ART = "因产品存在缺陷造成他人损害的，被侵权人可以向产品的生产者请求赔偿，也可以向产品的销售者请求赔偿。"


def rows(law, n):
    """造 n 条候选题目，都标同一个法名"""
    return [{"_law": law, "id": f"{law}-{i}", "条号": str(i)} for i in range(1, n + 1)]


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

    print(f"has_leak —— 问题里出现条文正文的连续 {LEAK_N} 字就算泄题")
    # 重合字数都是量出来的，不是估的（这两个字串分别是 13 字和 11 字）
    check("重合 13 个字（阈值之上）→ 判泄题",
          has_leak("因产品存在缺陷造成他人损害", ART), True)
    check("只重合 11 个字（阈值之下）→ 不算泄题，这是边界",
          has_leak("存在缺陷造成他人损害的", ART), False)
    check("问题里有标点，归一化之后仍然算泄题",
          has_leak("因产品存在缺陷，造成他人损害", ART), True)
    check("条文本身短于阈值 → 不判泄题（没法切出 n 字片段）",
          has_leak("因产品存在缺陷造成他人的", "这条太短了"), False)
    check("换成普通人说法 → 不判泄题",
          has_leak("买到的东西有质量问题该找谁赔", ART), False)
    check("阈值可以调小，用来扫更短的重合",
          has_leak("缺陷造成", ART, n=4), True)
    check("空问题 → 不判泄题", has_leak("", ART), False)

    # 归一化是泄题判断的前提：标点和空白不参与比对，否则"照抄但改了标点"就漏过去了
    check("norm 去掉标点和空白，只留字",
          norm("因产品存在缺陷，造成他人 损害的。"), "因产品存在缺陷造成他人损害的")

    print("\npick —— 按法分层等比分配名额")
    a, b = rows("医疗保障法", 100), rows("生态环境法典", 100)
    got = pick(a + b, 100)

    def law_count(selected, law):
        return sum(1 for c in selected if c["_law"] == law)

    check("两层等大、取 100 → 各 50",
          [law_count(got, "医疗保障法"), law_count(got, "生态环境法典")], [50, 50])
    check("总数正好等于要求的题量", len(got), 100)
    check("没有重复条目", len({c["id"] for c in got}), len(got))

    # 2026-09-21 那个坑：候选是按法分批生成的（先全部医疗保障法，再全部生态环境法典），
    # 写成 rows[:n] 就等于只取排最前面的那部法，50 题里 48 题是医疗保障法。
    got = pick(a + b, 50)
    check("候选按法分批排、取 50 → 两层都要有，不能只拿排最前面那部法",
          [law_count(got, "医疗保障法") > 0, law_count(got, "生态环境法典") > 0],
          [True, True])

    got = pick(rows("民法典", 100) + rows("司法解释", 300), 100)
    check("两层 1:3、取 100 → 按比例 25/75",
          [law_count(got, "民法典"), law_count(got, "司法解释")], [25, 75])

    picked = [c["id"] for c in pick(rows("民法典", 100), 10)]
    check("层内要打乱，不能总是取条号最小的那 10 条",
          picked == [f"民法典-{i}" for i in range(1, 11)], False)

    check("要的题量超过候选总数 → 全部返回，不报错",
          len(pick(a + b, 500)), 200)
    check("空候选 → 空结果", pick([], 10), [])
    check("要 0 题 → 空结果", pick(a, 0), [])

    # pick 是在传进来的列表上原地 shuffle 的：同一个 rows 对象连调两次，
    # 第二次是在已经打乱的基础上再打乱，结果会变。想可复现就得每次重新构造 rows。
    # 现在唯一的调用方（main）只调一次，所以没暴露出来。
    first = [c["id"] for c in pick(rows("民法典", 100), 10)]
    second = [c["id"] for c in pick(rows("民法典", 100), 10)]
    check("固定种子：重新构造同样的候选，取到的是同一批题", first, second)

    print(f"\n{ok}/{ok + bad} 通过")
    if bad:
        print(f"有 {bad} 个用例失败 —— 评测集的组成方式变了，已有数字可能不再可比")
        return 1
    print("筛选判据没变：既不会放泄题进来，也不会让评测集偏科")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
