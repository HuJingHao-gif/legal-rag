# -*- coding: utf-8 -*-
"""
多跳端到端评测：答案里**引用的覆盖**够不够，而不只是检索捞没捞到。

和 eval_answer.py 的区别在指标。那边是单跳题，判定"唯一引用是否精确命中"，
多引一条就算错（那是为了治"撒网列 6 条"故意收紧的）。多跳题要 2~3 条才答得全，
只让引一条就是结构上答不对，所以必须换成多引用 + 覆盖率。

三个指标要一起看，单看任何一个都会被玩坏：
  引用覆盖率 = |引用 ∩ 金标| / |金标|   平均答全了几成
  全中率     = 引用集合完全等于金标     不允许多引
  多引率     = 引了金标之外的条文       防"撒网"重新抬头
只看全中率的话，模型把金标全列上再顺手列 10 条无关的也能拿高分 —— 那就是最初踩过的坑。

用法:
  python src/eval_multihop_answer.py --eval data/多跳评测集.jsonl --mode bm25 --limit 20
  python src/eval_multihop_answer.py --eval data/交叉引用_评测集.jsonl --mode bm25 --agentic
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))
from answer import ask, ask_agentic
from textutil import cn2int, ART_RE
from eval_multihop import gold_of

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_FILE = DATA_DIR / "评测结果.jsonl"
LAW_RE = re.compile(r"《([^》]+)》")
WORKERS = 6


def cited_pairs(citations):
    """把模型给的引用字符串解析成 [(法名, 条号整数)]。

    引用写法是"《法律名称》第X条"，法名和条号都可能写成全称/简称、
    中文数字/阿拉伯数字，所以两边都归一化后再比。
    """
    out = []
    for c in citations or []:
        s = str(c)
        m_law = LAW_RE.search(s)
        m_art = ART_RE.search(s)
        if m_art:
            out.append((m_law.group(1) if m_law else "", cn2int(m_art.group(0))))
    return out


def law_hit(cited_law, gold_law):
    if not cited_law:
        return False
    return gold_law in cited_law or cited_law in gold_law


def score_one(citations, gold):
    """返回 (命中数, 金标数, 多引数)"""
    cited = cited_pairs(citations)
    used = set()
    hit = 0
    for i, (glaw, gno) in enumerate(gold):
        for cl, cn in cited:
            if cn == cn2int(gno) and law_hit(cl, glaw):
                hit += 1
                used.add(i)
                break
    return hit, len(gold), max(0, len(cited) - len(used))


def run_one(q, index, k, mode, use_rewrite, agentic, max_rounds):
    fn = ask_agentic if agentic else ask
    kwargs = dict(corpus=index, k=k, mode=mode, use_rewrite=use_rewrite, multi_cite=True)
    if agentic:
        kwargs["max_rounds"] = max_rounds
    try:
        r = fn(q["问题"], **kwargs)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    # ★ ask()/ask_agentic() 内部会把生成失败降级成一句"生成失败：…"，引用留空。
    #   不拦住的话，评测看到的就是"模型没给引用"，跟"模型答错了"长得一模一样 ——
    #   踩过：并发跑 agentic 时限流，92% 的题静默失败，差点当成真实结果。
    if str(r.get("answer", "")).startswith("生成失败"):
        return None, str(r["answer"])[:90]
    return r, None


def main(eval_file, index="all", k=5, mode="bm25", use_rewrite=True, agentic=False,
         max_rounds=3, limit=None, workers=WORKERS):
    questions = [json.loads(l) for l in open(eval_file, encoding="utf-8")]
    if limit:
        questions = questions[:limit]
    tag = "agentic" if agentic else "single"

    print(f"{Path(eval_file).name}｜{len(questions)} 题｜{tag}｜mode={mode} k={k} "
          f"｜multi_cite=开\n")

    results, errs = [], 0

    def work(q):
        return q, *run_one(q, index, k, mode, use_rewrite, agentic, max_rounds)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (q, r, err) in enumerate(ex.map(work, questions), 1):
            if err or r is None:
                errs += 1
            else:
                hit, n, extra = score_one(r.get("citations"), gold_of(q))
                results.append({"id": q["id"], "hit": hit, "n": n, "extra": extra,
                                "cites": r.get("citations")})
            if i % 20 == 0:
                print(f"  {i}/{len(questions)}", flush=True)

    if not results:
        raise SystemExit(f"全部失败，检查 API key / 并发数。样例错误：{errs or '无'}")
    if errs > 0.1 * len(questions):
        raise SystemExit(
            f"★ 失败率 {errs}/{len(questions)} 超过 10%，拒绝出结果 —— "
            "并发限流会伪装成'模型没给引用'，那种数字比没有更危险。"
            "调小 --workers 重跑。"
        )

    tot = len(results)
    cov = sum(r["hit"] / r["n"] for r in results) / tot
    full = sum(1 for r in results if r["hit"] == r["n"] and r["extra"] == 0) / tot
    anyp = sum(1 for r in results if r["hit"] > 0) / tot
    extra = sum(r["extra"] for r in results) / tot

    print("\n" + "=" * 62)
    print(f"  n={tot}｜API 失败 {errs}")
    print(f"  引用覆盖率（平均答全几成）: {cov:.1%}")
    print(f"  至少引中一条             : {anyp:.1%}")
    print(f"  ★ 全中率（不多不少）     : {full:.1%}")
    print(f"  多引条数（每题平均）     : {extra:.2f}")
    print("=" * 62)

    worst = sorted(results, key=lambda r: r["hit"] / r["n"])[:3]
    for r in worst:
        print(f"  差例 {r['id']}: 中 {r['hit']}/{r['n']}，引用 {r['cites']}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tag": f"multihop_answer_{tag}_{mode}_{Path(eval_file).stem}",
            "题数": tot, "引用覆盖率": round(cov, 4),
            "至少引中一条": round(anyp, 4), "全中率": round(full, 4),
            "多引条数": round(extra, 3),
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    ev = a[a.index("--eval") + 1] if "--eval" in a else str(DATA_DIR / "多跳评测集.jsonl")
    main(ev,
         index=a[a.index("--index") + 1] if "--index" in a else "all",
         k=int(a[a.index("--k") + 1]) if "--k" in a else 5,
         mode=a[a.index("--mode") + 1] if "--mode" in a else "bm25",
         use_rewrite="--no-rewrite" not in a,
         agentic="--agentic" in a,
         max_rounds=int(a[a.index("--rounds") + 1]) if "--rounds" in a else 3,
         limit=int(a[a.index("--limit") + 1]) if "--limit" in a else None,
         workers=int(a[a.index("--workers") + 1]) if "--workers" in a else WORKERS)
