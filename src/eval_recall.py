# -*- coding: utf-8 -*-
"""
检索评测：recall@1 / @5 / @10，支持多模式扫描 + 多索引。

判定口径定死：一题命中@k = 它的标准条号出现在检索结果的前 k 条里。

跨法条号撞车（用合并库之后必须处理）：
  "第一条"在民法典、4 部司法解释、2 部新法里都有，所以合并库下必须连法名一起比，
  只比条号会大量假命中。统一走 is_gold()：条号要对，法名也要对得上。

索引和评测集是解耦的：
  评测集决定"考什么题"，索引决定"在多小的库里找"。用 --index all 就能测
  "同一批题从单库换到合并库会掉多少" —— 那就是跨法干扰的真实成本。

用法:
  python src/eval_recall.py --sweep                   # 民法典题 · 民法典库
  python src/eval_recall.py --sweep laws              # 司法解释题 · 司法解释库
  python src/eval_recall.py --sweep laws --index all  # 司法解释题 · 合并库 ← 跨法干扰
  python src/eval_recall.py --sweep --rewrite new --index all
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (search, load_index, build_bm25,
                      CHUNKS, VECTORS, SFJS_CHUNKS, SFJS_VECTORS, NEW_CHUNKS, NEW_VECTORS,
                      ALL_CHUNKS, ALL_VECTORS)
import rewrite as rw

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_FILE = DATA_DIR / "评测结果.jsonl"

# 评测集：gold_law=None 表示"法名取题目自带的 备注 字段"
# 09-22 换成 _v2：v1 只有 38/50/50 题，噪声地板 ±5.5pp，比想测的效应还大。
# v2 = 每语料 100 题。注意单语料地板其实是 ±5.9pp，不是早先文档写的 ±2.2pp
# —— 那个数字把 300 当成了单语料样本量。v1 文件还在 data/ 里，没删。
EVAL_SETS = {
    "minfadian": {"eval": DATA_DIR / "评测集_v2.jsonl",           "gold_law": "民法典"},
    "laws":      {"eval": DATA_DIR / "司法解释_评测集_v2.jsonl",   "gold_law": None},
    "new":       {"eval": DATA_DIR / "新法_评测集_v2.jsonl",       "gold_law": None},
}

# 索引库
INDEXES = {
    "minfadian": (CHUNKS, VECTORS),
    "laws":      (SFJS_CHUNKS, SFJS_VECTORS),
    "new":       (NEW_CHUNKS, NEW_VECTORS),
    "all":       (ALL_CHUNKS, ALL_VECTORS),
}

KS = (1, 5, 10)
SWEEP = ["dense", "bm25", "hybrid"]
RERANK_SWEEP = ["dense", "dense+rerank", "hybrid", "hybrid+rerank"]


def is_gold(c, q, gold_law):
    """条号要对，法名也要对。法名用"包含"判断，容忍全称/简称/书名号的写法差异。"""
    if c["条号"] != q["标准条号"][0]:
        return False
    if not gold_law:
        return True
    c_law = c.get("简称") or c.get("law") or ""
    return gold_law in c_law or c_law in gold_law


def eval_mode(mode, questions, chunks, vectors, bm25, use_rewrite=False):
    hits  = defaultdict(lambda: {k: 0 for k in KS})
    total = defaultdict(int)
    failures = []

    for q in questions:
        gold_law = q.get("_gold_law")
        query = rw.expand(q["问题"]) if use_rewrite else q["问题"]
        results = search(query, k=max(KS), mode=mode,
                         chunks=chunks, vectors=vectors, bm25=bm25)
        rank = next((i + 1 for i, (_, c) in enumerate(results) if is_gold(c, q, gold_law)), None)
        total[q["类型"]] += 1
        for k in KS:
            if rank is not None and rank <= k:
                hits[q["类型"]][k] += 1
        if rank is None or rank > 10:
            failures.append(q["id"])
    return hits, total, failures


def main(modes, corpus="minfadian", use_rewrite=False, index=None):
    conf = EVAL_SETS[corpus]
    index = index or corpus
    questions = [json.loads(l) for l in open(conf["eval"], encoding="utf-8")]
    for q in questions:
        q["_gold_law"] = conf["gold_law"] or q.get("备注", "")

    chunks, vectors = load_index(*INDEXES[index])
    bm25 = build_bm25(chunks) if any(m.startswith(("bm25", "hybrid")) for m in modes) else None
    print(f"评测集 {corpus}｜题数 {len(questions)}｜索引 {index}（{len(chunks)} 条）"
          f"｜改写 {'开' if use_rewrite else '关'}")
    print(f"模式 {modes}\n")

    header = f"{'模式':<14}{'@1':>8}{'@5':>8}{'@10':>8}   │ 口语化 @1/@5/@10      │ 原话 @1/@5/@10"
    print("=" * len(header) + "═" * 10)
    print(header)
    print("-" * (len(header) + 10))

    for mode in modes:
        hits, total, failures = eval_mode(mode, questions, chunks, vectors, bm25, use_rewrite)
        n = len(questions)
        r = {k: sum(hits[t][k] for t in total) / n for k in KS}

        def cell(t, k):
            return f"{hits[t][k]/total[t]:.1%}" if total[t] else "—"
        ko = "/".join(cell("口语化", k) for k in KS)
        yq = "/".join(cell("法条原话", k) for k in KS)
        tag = f"{mode}{'+rw' if use_rewrite else ''}"
        print(f"{tag:<14}{r[1]:>8.1%}{r[5]:>8.1%}{r[10]:>8.1%}   │ {ko:<22}│ {yq}")
        print(f"{'':<14}失败题: {failures if failures else '无'}")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "tag": f"{tag}_{corpus}" + ("" if index == corpus else f"@{index}"),
                "题数": n,
                "recall@1": round(r[1], 4), "recall@5": round(r[5], 4),
                "recall@10": round(r[10], 4),
                "口语化": {f"@{k}": round(hits["口语化"][k] / total["口语化"], 4) for k in KS}
                          if total["口语化"] else None,
                "法条原话": {f"@{k}": round(hits["法条原话"][k] / total["法条原话"], 4) for k in KS}
                            if total["法条原话"] else None,
                "失败题": failures,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    corpus = next((a for a in args if a in EVAL_SETS), "minfadian")
    index = args[args.index("--index") + 1] if "--index" in args else None
    use_rewrite = "--rewrite" in args
    if "--rerank" in args:
        modes = RERANK_SWEEP
    elif "--sweep" in args:
        modes = SWEEP
    else:
        modes = [next((a for a in args if a not in EVAL_SETS and a not in INDEXES
                       and not a.startswith("--")), "dense")]
    main(modes, corpus, use_rewrite, index)
    print(f"\n已追加 {len(modes)} 行到 {LOG_FILE}")
