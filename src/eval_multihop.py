# -*- coding: utf-8 -*-
"""
多跳检索评测：一道题要 2~3 条法条**全部**被捞到，才算命中。

和 eval_recall.py 的区别就在这：
  那边的题是单条可答的（make_eval.py 明确要求"只靠这一条就能回答"），一题一个标准条号，
  指标是 recall@k；单轮检索在那种题上 recall@5 已经 99~100%，天花板到了，
  再优化也看不出区别。
  这里的题需要多条配合，指标是 hit_all@k（全部命中），才有区分度。

顺带也报 hit_any@k（至少中一条）和 mean_cov@k（平均覆盖率）：
只看 hit_all 会丢掉"差一条"和"一条没中"的区别，而那两种失败的治疗方式不一样。

用法:
  python src/eval_multihop.py --sweep --rewrite                 # 多跳集，三种模式扫描
  python src/eval_multihop.py --mode hybrid --rewrite --index all
  python src/eval_multihop.py --hop 3 --sweep --rewrite         # 只看三跳题
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
EVAL_FILE = DATA_DIR / "多跳评测集.jsonl"

INDEXES = {
    "minfadian": (CHUNKS, VECTORS),
    "laws":      (SFJS_CHUNKS, SFJS_VECTORS),
    "new":       (NEW_CHUNKS, NEW_VECTORS),
    "all":       (ALL_CHUNKS, ALL_VECTORS),
}
KS = (1, 5, 10)
SWEEP = ["dense", "bm25", "hybrid"]


def law_ok(c, gold_law):
    """合并库下必须连法名一起比。法名用包含判断，容忍全称/简称/书名号差异。"""
    if not gold_law:
        return True
    c_law = c.get("简称") or c.get("law") or ""
    return gold_law in c_law or c_law in gold_law


def gold_of(q):
    """统一取金标，返回 [(法名, 条号), ...]。

    两套多跳评测集的格式不一样，这里抹平：
      随机跨章那套："标准法律" + "标准条号"，金标同属一部法；
      交叉引用那套："金标" 列表，每条自带法名 —— 因为它跨了两部法
      （司法解释 + 民法典），没法用一个"标准法律"字段表达。
    """
    if q.get("金标"):
        return [(g["法"], g["条号"]) for g in q["金标"]]
    law = q.get("_gold_law") or q.get("标准法律", "")
    return [(law, n) for n in (q.get("标准条号") or [])]


def hit_set(gold_pairs, results, k):
    """前 k 条里命中了哪些金标。条号要对，法名也要对（合并库下必须）。"""
    got = set()
    for _, c in results[:k]:
        c_law = c.get("简称") or c.get("law") or ""
        for law, no in gold_pairs:
            if c["条号"] == no and (law in c_law or c_law in law):
                got.add((law, no))
    return got


def eval_mode(mode, questions, chunks, vectors, bm25, use_rewrite=False):
    """跑一遍，按 k 和跳数分别累计。

    返回 stat[k][bucket] = {all, any, cov, n}，bucket 是 "__all__" 或 "2跳"/"3跳"。
    """
    stat = {k: defaultdict(lambda: {"all": 0, "any": 0, "cov": 0.0, "n": 0}) for k in KS}
    failures = []

    for q in questions:
        gold = gold_of(q)
        hop = len(gold)
        query = rw.expand(q["问题"]) if use_rewrite else q["问题"]
        results = search(query, k=max(KS), mode=mode,
                         chunks=chunks, vectors=vectors, bm25=bm25)

        per_k = {k: hit_set(gold, results, k) for k in KS}
        for k in KS:
            got = per_k[k]
            cov = len(got) / hop
            for bucket in ("__all__", f"{hop}跳"):
                s = stat[k][bucket]
                s["n"] += 1
                s["cov"] += cov
                s["all"] += (cov == 1.0)
                s["any"] += (len(got) > 0)

        if len(per_k[10]) < hop:
            failures.append(f"{q['id']}(缺{hop - len(per_k[10])})")

    return stat, failures


def eval_agentic(questions, index="all", k=5, mode="hybrid", use_rewrite=True,
                 max_rounds=3, workers=8):
    """多轮检索：报"每一轮累计上下文"下的 hit_all。

    ★ 比公平性：多轮会把上下文撑大（3 轮最多 k×3 条）。所以单轮要拿**同样大小的
    上下文**来对照（见 main 里打印的 single_k 行），否则是拿 15 条比 5 条，赢了也不说明问题。
    """
    from concurrent.futures import ThreadPoolExecutor
    from answer import retrieve_agentic       # 跟产品走同一份实现，别写第二遍

    def run(q):
        rounds, _ = retrieve_agentic(q["问题"], corpus=index, k=k, mode=mode,
                                     use_rewrite=use_rewrite, max_rounds=max_rounds)
        return q, rounds

    # 每题要调 max_rounds-1 次判定，串行太慢，这里并发跑
    done = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (q, rounds) in enumerate(ex.map(run, questions), 1):
            done.append((q, rounds))
            if i % 20 == 0:
                print(f"  多轮检索 {i}/{len(questions)}", flush=True)

    stat = {r: {"all": 0, "any": 0, "cov": 0.0, "n": 0, "size": 0}
            for r in range(max_rounds)}
    used = []
    for q, rounds in done:
        gold = gold_of(q)
        hop = len(gold)
        used.append(len(rounds))
        for r in range(max_rounds):
            cur = rounds[min(r, len(rounds) - 1)]   # 早停的题，后续轮次沿用最后一次累计
            got = hit_set(gold, cur, len(cur))
            cov = len(got) / hop
            s = stat[r]
            s["n"] += 1
            s["cov"] += cov
            s["size"] += len(cur)
            s["all"] += (cov == 1.0)
            s["any"] += (len(got) > 0)

    return stat, used


def single_turn_at(questions, chunks, vectors, bm25, ks, mode, use_rewrite):
    """单轮在若干 k 上的 hit_all，给 agentic 做对照"""
    out = {k: {"all": 0, "n": 0} for k in ks}
    for q in questions:
        gold = gold_of(q)
        hop = len(gold)
        query = rw.expand(q["问题"]) if use_rewrite else q["问题"]
        results = search(query, k=max(ks), mode=mode,
                         chunks=chunks, vectors=vectors, bm25=bm25)
        for k in ks:
            got = hit_set(gold, results, k)
            out[k]["n"] += 1
            out[k]["all"] += (len(got) == hop)
    return {k: v["all"] / v["n"] for k, v in out.items()}


def main(modes, use_rewrite=False, index="all", hop_filter=None, agentic=False,
         max_rounds=3, k=5, eval_file=None):
    path = Path(eval_file) if eval_file else EVAL_FILE
    questions = [json.loads(l) for l in open(path, encoding="utf-8")]
    for q in questions:
        q["_gold_law"] = q.get("标准法律", "")     # 交叉引用那套没这字段，交给 gold_of()
    if hop_filter:
        questions = [q for q in questions if len(gold_of(q)) == hop_filter]

    chunks, vectors = load_index(*INDEXES[index])
    bm25 = build_bm25(chunks) if any(m.startswith(("bm25", "hybrid")) for m in modes) else None

    hops = defaultdict(int)
    for q in questions:
        hops[len(gold_of(q))] += 1
    print(f"{path.name}｜{len(questions)} 题（{dict(hops)}）｜索引 {index}（{len(chunks)} 条）"
          f"｜改写 {'开' if use_rewrite else '关'}\n")

    header = (f"{'模式':<14}{'hit_all@1':>11}{'hit_all@5':>11}{'hit_all@10':>12}"
              f"{'hit_any@5':>11}{'mean_cov@5':>12}")
    print("=" * 82)
    print(header)
    print("-" * 82)

    all_res = {}
    for mode in modes:
        res, failures = eval_mode(mode, questions, chunks, vectors, bm25, use_rewrite)
        all_res[mode] = res
        tag = f"{mode}{'+rw' if use_rewrite else ''}"
        row = []
        for kk in KS:          # ★ 别用 k：循环变量会泄漏到函数作用域，把参数 k 冲掉（踩过）
            s = res[kk]["__all__"]
            row.append(s["all"] / s["n"] if s["n"] else 0)
        s5 = res[5]["__all__"]
        print(f"{tag:<14}{row[0]:>10.1%}{row[1]:>11.1%}{row[2]:>12.1%}"
              f"{s5['any']/s5['n']:>11.1%}{s5['cov']/s5['n']:>12.2f}")
        if failures:
            print(f"{'':<14}@10 仍缺条的题: {len(failures)} 道 {failures[:6]}")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "tag": f"multihop_{tag}@{index}",
                "题数": len(questions),
                "hit_all@1": round(row[0], 4), "hit_all@5": round(row[1], 4),
                "hit_all@10": round(row[2], 4),
                "hit_any@5": round(s5["any"] / s5["n"], 4) if s5["n"] else None,
                "mean_cov@5": round(s5["cov"] / s5["n"], 4) if s5["n"] else None,
            }, ensure_ascii=False) + "\n")

    print("=" * 82)
    print("\n按跳数拆分（看 hit_all@5）：")
    for mode in modes:
        res = all_res[mode]
        parts = []
        for bucket in sorted(b for b in res[5] if b != "__all__"):
            s = res[5][bucket]
            parts.append(f"{bucket} n={s['n']} all@5={s['all']/s['n']:.1%}")
        print(f"  {mode}{'+rw' if use_rewrite else ''}: " + " | ".join(parts))

    if agentic:
        m = modes[0]
        stat, used = eval_agentic(questions, index, k, m, use_rewrite, max_rounds)
        sizes = [stat[r]["size"] / stat[r]["n"] for r in range(max_rounds)]
        ks = sorted({max(1, int(round(s))) for s in sizes})
        single = single_turn_at(questions, chunks, vectors, bm25, ks, m, use_rewrite)

        print("=" * 82)
        print(f"\n多轮检索（{m}{'+rw' if use_rewrite else ''}，每轮 k={k}，最多 {max_rounds} 轮）")
        print(f"{'':<10}{'累计条数':>10}{'hit_all':>10}{'hit_any':>10}{'mean_cov':>10}   │ 单轮同尺寸对照")
        for r in range(max_rounds):
            s, n = stat[r], stat[r]["n"]
            size = s["size"] / n
            near = min(ks, key=lambda x: abs(x - size))
            print(f"  第{r+1}轮{'':<5}{size:>10.1f}{s['all']/n:>10.1%}{s['any']/n:>10.1%}"
                  f"{s['cov']/n:>10.2f}   │ k={near} → {single[near]:.1%}")
        dist = {r: used.count(r) for r in sorted(set(used))}
        print(f"  平均用轮数 {sum(used)/len(used):.2f}｜轮数分布 {dist}")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "tag": f"multihop_agentic_{m}@{index}",
                "题数": len(questions), "每轮k": k, "最大轮数": max_rounds,
                "各轮hit_all": [round(stat[r]["all"] / stat[r]["n"], 4) for r in range(max_rounds)],
                "各轮累计条数": [round(sizes[r], 2) for r in range(max_rounds)],
                "单轮同尺寸对照": {f"k={kk}": round(v, 4) for kk, v in single.items()},
                "平均用轮数": round(sum(used) / len(used), 3),
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    index = args[args.index("--index") + 1] if "--index" in args else "all"
    use_rewrite = "--rewrite" in args
    hop = int(args[args.index("--hop") + 1]) if "--hop" in args else None
    k = int(args[args.index("--k") + 1]) if "--k" in args else 5
    rounds = int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3
    ev = args[args.index("--eval") + 1] if "--eval" in args else None
    if "--sweep" in args:
        modes = SWEEP
    else:
        modes = [args[args.index("--mode") + 1]] if "--mode" in args else ["hybrid"]
    main(modes, use_rewrite, index, hop, agentic="--agentic" in args,
         max_rounds=rounds, k=k, eval_file=ev)
