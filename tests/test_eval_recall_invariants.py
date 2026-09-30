# -*- coding: utf-8 -*-
"""
评测集与语料的一致性测试：is_gold 的字符串相等假设成不成立。

为什么单独测它：is_gold 比条号用的是**原始字符串相等**
（c["条号"] != q["标准条号"][0]），比的是"第一千二百五十四条"这个串本身。
于是它隐含了一个假设：评测集里的标准条号，必须在语料里原样出现。

假设破了不会报错，只会静默变低 —— 那几道题永远命中不了，
表上掉的那几个点会被当成"检索变差了"，查错方向从一开始就偏。
最可能踩的一种：标准条号写成"第1254条"，而语料里是"第一千二百五十四条"。
这两种写法在 cn2int 眼里是同一条，在 is_gold 眼里是两个不相干的字符串。

这里只查"条号在语料里原样存在吗"。is_gold 法名包含判断本身的漏洞
（空法名放行、全称误命中）在 tests/test_recall_judgement.py 里，两处职责别混。

运行: python tests/test_eval_recall_invariants.py
"""
import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# 语料在哪。合并库也要查：它是 --index all 实际用的那份，
# 而"忘了重跑索引"是这项目踩过的坑（retrieve.py 开头记了行数校验那次）。
CORPUS = {
    "data/民法典_chunks.jsonl":        DATA / "民法典_chunks.jsonl",
    "data/laws/司法解释_chunks.jsonl":  DATA / "laws" / "司法解释_chunks.jsonl",
    "data/laws/新法_chunks.jsonl":      DATA / "laws" / "新法_chunks.jsonl",
    "data/all_chunks.jsonl":           DATA / "all_chunks.jsonl",
}

# 每个评测集怎么取金标：(固定法名, 取法名的字段, 说明)。
# 固定法名非 None 就直接用；否则从 field 里取。
# 民法典那几套没有法名字段（备注 存的是"第一编　总则 第一章…"），只能写死。
#
# ★ 这份表跟 eval_recall.EVAL_SETS 是两份独立硬编码，但**漏不了**：
#   main() 里有道 glob 闸门，data/ 下任何 *评测集*.jsonl 没登记就直接判失败。
#   所以新增评测集时不会"悄悄漏查"，只会红 —— 也就不依赖"记得改两处"。
EVAL_SETS = {
    "评测集.jsonl":           ("民法典", None, "v1 · 民法典题"),
    "评测集_v2.jsonl":        ("民法典", None, "v2 · eval_recall 当前在用"),
    "评测集_候选.jsonl":       ("民法典", None, "候选池，含被筛掉的题"),
    "司法解释_评测集.jsonl":    (None, "备注", "v1 · 司法解释题"),
    "司法解释_评测集_v2.jsonl": (None, "备注", "v2 · eval_recall 当前在用"),
    "新法_评测集.jsonl":       (None, "备注", "v1 · 新法题"),
    "新法_评测集_v2.jsonl":    (None, "备注", "v2 · eval_recall 当前在用"),
    "多跳评测集.jsonl":        (None, "标准法律", "多跳题 · 标准条号有多条"),
    "交叉引用_评测集.jsonl":   (None, None, "跨法多跳 · 金标是 [{法,条号}]，每条自带法名"),
}

# 名字里带"评测集"但确实不归本测试管的，在这里显式登记理由。
# 目前是空的 —— 交叉引用曾经被放进来过，理由是"不走 is_gold"。那个理由站不住：
# eval_multihop 的命中判据比的**也是字符串相等**（c["条号"] == 金标条号），
# 写法一漂那几道题就永远命中不了，表现成"检索变差了"。同一个坑，所以收回来查。
NOT_IS_GOLD = {}


def gold_pairs(q, fixed, field):
    """统一取 [(法名, 条号)]。

    两套多跳集的金标格式不一样：随机跨章那套用 标准法律 + 标准条号；
    交叉引用那套用 金标（每条自带法名，因为它跨了两部法）。
    """
    if q.get("金标"):
        return [(g.get("法", ""), g.get("条号", "")) for g in q["金标"]]
    law = fixed if fixed else q.get(field, "")
    return [(law, no) for no in (q.get("标准条号") or [])]


def load_corpus():
    """读语料，返回 {文件名: (法名集合, 条号集合)}。

    顺带说明为什么法名这里用**精确相等**、而不用 is_gold 的包含判断：
    司法解释的全称里含"民法典"三个字，用包含判断会把司法解释的 chunks
    算进民法典的题里，于是一道正常的民法典题会因为"司法解释库里没有这条"
    而误报。查一致性要的是精确定位"这部法的语料文件"，不是复刻判据。
    """
    out = {}
    for name, path in CORPUS.items():
        if not path.exists():
            raise SystemExit(
                f"语料文件不存在：{path}\n"
                f"这条不能跳过 —— 跳过了测试会显示全绿，但实际什么都没查。"
            )
        laws, nos, n = set(), set(), 0
        for line in open(path, encoding="utf-8"):
            c = json.loads(line)
            laws.add(c.get("简称") or "")
            laws.add(c.get("law") or "")
            nos.add(c["条号"])
            n += 1
        if not n:
            raise SystemExit(f"语料文件是空的：{path}")
        out[name] = (laws, nos, n)
    return out


def check_one_file(name, spec, index):
    """返回这个文件里所有的问题（人类可读的一行一条）。没问题就返回空列表。"""
    fixed, field, _ = spec
    path = DATA / name
    if not path.exists():
        raise SystemExit(
            f"评测集文件不存在：{path}\n"
            f"测试登记了它就必须查，静默跳过等于把这道防线删了。"
        )

    problems = []
    lines = [l for l in open(path, encoding="utf-8") if l.strip()]
    if not lines:
        raise SystemExit(f"评测集是空的：{path} —— 解析口径变了？空文件全绿没有意义")

    for i, line in enumerate(lines, 1):
        q = json.loads(line)
        qid = q.get("id") or f"第{i}行"          # 兜底：万一某套题没写 id

        pairs = gold_pairs(q, fixed, field)
        if not pairs:
            problems.append(f"{qid} | 取不到任何金标（字段 {field} / 金标）")
            continue

        for law, no in pairs:
            if not law:
                problems.append(f"{qid} | 金标里取不到法名（字段 {field}）")
                continue
            if not no:
                problems.append(f"{qid} | 金标里取不到条号")
                continue

            files = [k for k, (laws, _, _) in index.items() if law in laws]
            if not files:
                problems.append(
                    f"{qid} | 法名 {law!r} 在 4 个语料文件里都精确匹配不到 —— "
                    f"语料没摄入这部法，或题目的法名写法变了"
                )
                continue

            if not any(no in index[k][1] for k in files):
                problems.append(
                    f"{qid} | 法名 {law!r} | 金标条号 {no!r} 在这几个文件里都不是原样：{files} —— "
                    f"命中判据比的是字符串，这题永远命中不了（分数会静默偏低）"
                )
    return problems


def main():
    ok = bad = 0

    def check(desc, got, want):
        nonlocal ok, bad
        if got == want:
            ok += 1
            print(f"  ✓ {desc}")
            return
        bad += 1
        print(f"  ✗ {desc}")
        if isinstance(got, list) and got:
            for line in got[:20]:
                print(f"      {line}")
            if len(got) > 20:
                print(f"      …还有 {len(got) - 20} 条")
        else:
            print(f"      期望 {want}，实际 {got}")

    # 先确认没有"名字像评测集但没登记"的文件：那就是静默漏查。
    found = sorted(p.name for p in DATA.glob("*评测集*.jsonl"))
    if not found:
        print(f"在 {DATA} 下没 glob 到 *评测集*.jsonl —— 路径不对，或数据被挪走了")
        return 1
    unknown = [f for f in found if f not in EVAL_SETS and f not in NOT_IS_GOLD]
    check(f"data/ 下 {len(found)} 个 *评测集*.jsonl 全部已登记（没登记的会被漏查）",
          unknown, [])
    if unknown:
        print("      新评测集要登记进 EVAL_SETS，或者在 NOT_IS_GOLD 里写清为什么不走 is_gold；")
        print("      没登记的会被漏查，所以这里直接算失败：")
        for f in unknown:
            keys = sorted(json.loads(open(DATA / f, encoding="utf-8").readline()).keys())
            print(f"        {f}：字段 {keys}")

    index = load_corpus()
    # 报的是每个文件的实际行数，不是"去重条号数"—— 后者跨法重复，
    # 加在一起会得到一个人为的、任何文件里都不存在的数字。
    sizes = " | ".join(f"{Path(k).name} {n} 条" for k, (_, _, n) in index.items())
    laws_all = {x for laws, _, _ in index.values() for x in laws if x}
    print(f"\n语料 {len(index)} 个文件：{sizes}（共 {len(laws_all)} 个法名）")

    print("\nis_gold 的字符串相等假设：标准条号必须在语料里原样存在")
    total_q = 0
    for name, spec in EVAL_SETS.items():
        n = len([l for l in open(DATA / name, encoding="utf-8") if l.strip()])
        total_q += n
        check(f"{name}（{spec[2]}，{n} 题）标准条号都能按法名原样找到",
              check_one_file(name, spec, index), [])

    if NOT_IS_GOLD:
        print("\n明确排除的文件：断言它的形状没变，免得它变成走金标的题还被排除")
        for name, reason in NOT_IS_GOLD.items():
            path = DATA / name
            if not path.exists():
                print(f"  ✗ 排除项文件不存在：{path}（{reason}）")
                bad += 1
                continue
            first = json.loads(open(path, encoding="utf-8").readline())
            check(f"{name} 仍然没有 标准条号（{reason}）", "标准条号" in first, False)

    print(f"\n共查 {total_q} 道题 | {ok}/{ok + bad} 通过")
    if bad:
        print("有检查没通过 —— 评测集和语料对不上了，recall 数字已经不可信")
        return 1
    print("评测集的标准条号在语料里都是原样存在的，is_gold 的字符串比较没问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
