# -*- coding: utf-8 -*-
"""
多跳候选题筛选：候选 → 定稿多跳评测集。

比 filter_eval.py 多两道闸门，因为多跳题的"假货"更难一眼看出来：
  ① 泄题要对**每一条**金标条文都查一遍 —— 只要有一条的正文片段出现在问题里，
     那道题就废了（用户把答案念出来了）。
  ② 每条金标必须**有独家贡献**。生成时模型要逐条说明"这条独家提供了什么"，
     写不出来的说明它是凑数的，整题丢弃。这是防"总则+分则重复规定同一件事"
     那种假多跳的主要手段。

另外校验每条金标条文确实还在语料库里 —— 不在的话评测会静默算成 0 分。

用法: python data/filter_multihop.py [候选.jsonl] [chunks.jsonl] [输出.jsonl] [定稿题量]
"""
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from filter_eval import norm, has_leak, LEAK_N, MIN_LEN, MAX_LEN

DATA_DIR = Path(__file__).parent
CAND = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "多跳_候选.jsonl"
CHUNKS = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "all_chunks.jsonl"
DST = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA_DIR / "多跳评测集.jsonl"
TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 100

MIN_ROLE = 10        # "独家信息"至少要写这么多字，否则视为没说出个所以然
MAX_PER_GOLDSET = 1  # 同一组金标条文只留一道题，防止换个问法反复考同一件事


def gold_of(c):
    """统一取金标 [(法名, 条号)]，兼容两套格式（跟 src/eval_multihop.gold_of 保持一致）：
    随机跨章那套用 标准法律+标准条号；交叉引用那套用 金标（每条自带法名，因为跨了两部法）。"""
    if c.get("金标"):
        return [(g["法"], g["条号"]) for g in c["金标"]]
    law = c.get("标准法律", "")
    return [(law, n) for n in (c.get("标准条号") or [])]


def main():
    cands = [json.loads(l) for l in open(CAND, encoding="utf-8")]
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    body = {(c.get("简称") or c.get("law"), c["条号"]): c["正文"] for c in chunks}

    kept, drop = [], Counter()
    seen_q, seen_gold = set(), set()

    for c in cands:
        q = c.get("问题", "").strip()
        gold = gold_of(c)

        if not (MIN_LEN <= len(q) <= MAX_LEN):
            drop["长度"] += 1
            continue
        if len(gold) < 2:
            drop["不足两跳"] += 1
            continue

        # 金标条文必须还在库里（法名对不上就按条号再试一次，容忍简称/全称差异）
        missing = [g for g in gold
                   if g not in body and not any(k[1] == g[1] for k in body)]
        if missing:
            drop["金标不在库"] += 1
            continue

        # 泄题：对**每一条**金标都查，有一条的正文片段出现在问题里就废
        leaked = False
        for g in gold:
            art = body.get(g) or next((v for k, v in body.items() if k[1] == g[1]), "")
            if art and has_leak(q, art):
                leaked = True
                break
        if leaked:
            drop["泄题"] += 1
            continue

        # 每条金标都得说出独家贡献 —— 防"总则+分则重复规定同一件事"的假多跳
        roles = {(str(r.get("法") or ""), str(r.get("条号"))): str(r.get("独家信息") or "")
                 for r in (c.get("每条的作用") or [])}
        if any(len(roles.get(g) or roles.get(("", g[1])) or "") < MIN_ROLE for g in gold):
            drop["说不出独家贡献"] += 1
            continue

        nq = norm(q)
        gkey = tuple(sorted(gold))
        if nq in seen_q or gkey in seen_gold:
            drop["重复"] += 1
            continue
        seen_q.add(nq)
        seen_gold.add(gkey)
        kept.append(c)

    # 配额：按跳数尽量均衡（2 跳和 3 跳都要有，别全是 2 跳）
    by_hop = defaultdict(list)
    for c in kept:
        by_hop[len(gold_of(c))].append(c)
    total = sum(len(v) for v in by_hop.values())
    final = []
    for hop, items in sorted(by_hop.items()):
        quota = round(TARGET * len(items) / total) if total else 0
        final += items[:quota]

    final.sort(key=lambda c: (len(gold_of(c)), gold_of(c)[0][0], gold_of(c)[0][1]))
    for i, c in enumerate(final, 1):
        c["id"] = f"mh{i:03d}"

    with open(DST, "w", encoding="utf-8") as f:
        for c in final:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"候选 {len(cands)} → 通过 {len(kept)} → 定稿 {len(final)}")
    print(f"  剔除：{dict(drop)}")
    print(f"  跳数分布：{dict(Counter(len(gold_of(c)) for c in final))}")
    print(f"  金标涉及的法：{dict(Counter(g[0] for c in final for g in gold_of(c)))}")
    print(f"已存 {DST}")


if __name__ == "__main__":
    main()
