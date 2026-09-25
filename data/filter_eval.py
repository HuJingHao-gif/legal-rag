# -*- coding: utf-8 -*-
"""
候选问题 → 定稿评测集。本来要人工筛，规则客观可复核，就自动化了。

筛四类：
  泄题：问题里出现条文正文的连续片段（>= LEAK_N 字），答案已经写在问题里
  重复：问题文字几乎一样，或同一个 (条号, 题型) 出现多次
  过短/过长：不像真人问法
  配额：题型 6:4，每部法都要有

用法: python data/filter_eval.py <候选.jsonl> <chunks.jsonl> <输出.jsonl> [定稿题量]
"""
import re
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

LEAK_N = 12      # 问题里连续 12 个字跟条文正文重合，判为泄题
                 # 只对"口语化"题查泄题。"法条原话"题本来就照抄条文，
                 # 对它查泄题会把整类杀光，实测 33 条全被误杀。
MIN_LEN, MAX_LEN = 8, 140
KO_RATIO = 0.6      # 口语化占比


# 归一化时要去掉的标点（用三重引号，避免里面的引号把字符串截断）
_PUNCT = re.compile(r"""[\s，。？、；：（）《》()【】"'"'\.\,\?\!\:\;]""")


def norm(s):
    """归一化：去掉空白和标点，只留字，用于查重"""
    return _PUNCT.sub("", s)


def has_leak(question, article, n=LEAK_N):
    """问题里是否含条文正文的连续 n 字片段"""
    q = norm(question)
    a = norm(article)
    if len(a) < n:
        return False
    grams = {a[i:i + n] for i in range(len(a) - n + 1)}
    return any(q[i:i + n] in grams for i in range(len(q) - n + 1))


def pick(rows, n, seed=42):
    """按 _law 分层等比分配名额选 n 条，层内固定种子随机取。

    不能写成 rows[:n]（2026-09-21 踩到的坑）：候选是按法分批生成的，
    取前 n 条等于只取排最前面的那部法。新法那次 50 题里 48 题全是医疗保障法，
    生态环境法典只剩 2 题。评测集一偏，数字就不代表整个库了。
    """
    if n <= 0:
        return []
    rnd = random.Random(seed)
    by = defaultdict(list)
    for c in rows:
        by[c["_law"]].append(c)
    for v in by.values():
        rnd.shuffle(v)                 # 层内打乱，否则总是取条号最小的那几条

    total = len(rows)
    quota = {k: n * len(v) / total for k, v in by.items()}   # 理想名额（小数）
    take = {k: int(q) for k, q in quota.items()}
    # 最大余数法：把剩下的名额补给"小数部分最大"的层，保证总数正好是 n
    left = n - sum(take.values())
    for k in sorted(by, key=lambda k: (-(quota[k] - take[k]), k))[:left]:
        take[k] += 1

    out = []
    for k, v in by.items():
        out += v[: min(take[k], len(v))]
    return out


def main(cand_path, chunks_path, out_path, total_target=50):
    cands = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    body = {(c.get("简称"), c["条号"]): c["正文"] for c in chunks}
    law_of = {c["条号"]: c.get("简称") for c in chunks}

    kept, seen_q, seen_pair, drop = [], set(), set(), {"泄题": 0, "重复": 0, "长度": 0}
    sample_leak = []
    for c in cands:
        q = c["问题"]
        if not (MIN_LEN <= len(q) <= MAX_LEN):
            drop["长度"] += 1
            continue
        law = c.get("备注") or law_of.get(c["标准条号"][0], "")
        art = body.get((law, c["标准条号"][0]), "")
        # 只查"口语化"题；"法条原话"本来就照抄条文
        if art and c["类型"] == "口语化" and has_leak(q, art):
            drop["泄题"] += 1
            if len(sample_leak) < 3:
                sample_leak.append((q, art))
            continue
        nq = norm(q)
        pair = (c["标准条号"][0], c["类型"], law)
        if nq in seen_q or pair in seen_pair:
            drop["重复"] += 1
            continue
        seen_q.add(nq); seen_pair.add(pair)
        c["_law"] = law
        kept.append(c)

    for q, art in sample_leak:
        print(f"  [泄题样例] 问题：{q[:50]}\n             条文：{art[:50]}")

    # 配额：口语化 : 法条原话 ≈ 6:4，每部法按比例拿名额
    ko = [c for c in kept if c["类型"] == "口语化"]
    yq = [c for c in kept if c["类型"] == "法条原话"]
    n_ko = min(len(ko), round(total_target * KO_RATIO))
    n_yq = min(len(yq), total_target - n_ko)
    final = pick(ko, n_ko) + pick(yq, n_yq)
    final.sort(key=lambda c: (c["_law"], c["标准条号"][0], c["类型"]))

    for i, c in enumerate(final, 1):
        c["id"] = f"q{i:03d}"

    out = Path(out_path)
    with open(out, "w", encoding="utf-8") as f:
        for c in final:
            c.pop("_law", None)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"候选 {len(cands)} → 通过 {len(kept)} → 定稿 {len(final)}")
    print(f"  剔除：{drop}")
    from collections import Counter
    print(f"  类型：{dict(Counter(c['类型'] for c in final))}")
    print(f"  按法：{dict(Counter(c.get('备注','') for c in final))}")
    print(f"已存 {out}")


if __name__ == "__main__":
    # 第 4 个参数是定稿题量（默认 50）；扩样本时按法传，例如 100
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    main(sys.argv[1], sys.argv[2], sys.argv[3], target)
