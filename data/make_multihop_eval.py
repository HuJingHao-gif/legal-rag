# -*- coding: utf-8 -*-
"""
多跳评测集生成：题目需要 2~3 条法条**互相配合**才能回答。

为什么单独做一套：现有评测集的题是刻意设计成"单条可答"的（见 make_eval.py 的提示词
"每个问题要只靠这一条就能回答"），单轮检索在那种题上 recall@5 已经 99~100%，
没有提升空间。要验证 agentic 多轮检索，得先有它治得了的病。

抽样策略：**同一部法、不同章**里各抽一条组成一束（默认 12 条）。
同一部法保证话题有交集、能凑出真问题；跨章强制多跳，避免"总则+分则重复规定
同一件事"那种假多跳。

用法:
  python data/make_multihop_eval.py                       # 默认 40 束
  python data/make_multihop_eval.py <束数> <输出.jsonl>
输出: 候选多跳题，交给 filter_multihop.py 筛
"""
import sys
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import llm

DATA_DIR = Path(__file__).parent
SRC = DATA_DIR / "all_chunks.jsonl"
DST = DATA_DIR / "多跳_候选.jsonl"

N_BUNDLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
if len(sys.argv) > 2:
    DST = Path(sys.argv[2])

BUNDLE = 12       # 每束塞几条待选条文
WORKERS = 6       # 并发数（纯 API 调用，线程池就够）
MAX_PER_BUNDLE = 3   # 一束最多产出几道题

PROMPT = """你是法律问答数据集的设计者。下面给你一组中国法条文，它们来自**同一部法的不同章节**。

请从中挑出**需要互相配合才能回答**的 2~3 条，设计一个**普通人真会问**的问题。

硬要求：
1. 问题必须**同时依赖**你挑的每一条 —— 只看其中任意一条都答不完整。
   如果两条只是在不同章节重复规定同一件事，那是**假多跳，不要**。
2. 用日常口语问，别照搬条文用语，不许出现"第X条"。
3. 对挑中的每一条，说明它**独家**提供了什么信息（别的条给不了的）。
   写不出独家信息的，说明它是多余的，别挑它。

只输出 JSON 数组，不要任何解释：
[{{"问题": "...",
  "条号": ["第X条", "第Y条"],
  "每条的作用": [{{"条号": "第X条", "独家信息": "..."}}, {{"条号": "第Y条", "独家信息": "..."}}]}}]

**如果这组条文凑不出合格的多跳题，就返回空数组 []**。宁缺毋滥 —— 大部分组合确实凑不出来，这很正常。

待选条文：
{listing}
"""


def load_chunks():
    return [json.loads(l) for l in open(SRC, encoding="utf-8")]


def stratify(chunks):
    """按 (法名, 章) 分层。章为空的（司法解释只有一个层级）就按 (法名, 条号区间) 粗分。"""
    by = defaultdict(list)
    for c in chunks:
        law = c.get("简称") or c.get("law") or "?"
        章 = c.get("章") or c.get("编") or ""
        if not 章:
            # 没有章节信息的，用条号除以 50 当粗粒度桶，保证一束里条文来自不同区段
            try:
                from textutil import cn2int
                bucket = cn2int(c["条号"]) // 50
            except Exception:
                bucket = 0
            章 = f"区间{bucket}"
        by[(law, 章)].append(c)
    return by


def make_bundles(chunks, n):
    """每束 = 同一部法、尽量不同章的各一条。

    只用"层次够多"的法：层数少的话一束里凑不出真正跨章的条文，
    会退化成强行配对（实测：1542/2802 条没有章节信息，那几部法各只有 1~2 层）。
    """
    by = stratify(chunks)
    by_law = defaultdict(list)
    for (law, _), items in by.items():
        by_law[law].append(items)

    eligible = {law: g for law, g in by_law.items() if len(g) >= 5}
    if not eligible:
        return [], {}
    laws = list(eligible)
    weights = [len(eligible[l]) for l in laws]      # 章多的法多抽，别让 80 章和 25 章等权

    rnd = random.Random(42)          # 固定种子，抽样可复现
    bundles = []
    for _ in range(n):
        law = rnd.choices(laws, weights=weights, k=1)[0]
        groups = eligible[law]
        picked = [rnd.choice(g) for g in rnd.sample(groups, min(BUNDLE, len(groups)))]
        bundles.append(picked)
    return bundles, eligible


def ask_bundle(bundle):
    """一束条文 → 至多 MAX_PER_BUNDLE 道题"""
    listing = "\n\n".join(
        f'[{i}] {c["条号"]}｜《{c.get("简称") or c.get("law")}》'
        + (f'｜{c.get("章")}' if c.get("章") else "")
        + f'\n{c["正文"]}'
        for i, c in enumerate(bundle)
    )
    out = llm.ask_json(PROMPT.format(listing=listing))
    if not isinstance(out, list):
        return []
    return out[:MAX_PER_BUNDLE]


def main():
    chunks = load_chunks()
    bundles, eligible = make_bundles(chunks, N_BUNDLES)
    if not bundles:
        raise SystemExit("没有层次足够的语料，先检查stratify()")
    print(f"语料 {len(chunks)} 条｜可用（层次≥5）的法：{ {k: len(v) for k, v in eligible.items()} }")
    print(f"→ 组出 {len(bundles)} 束（每束 {BUNDLE} 条，同法不同章）")

    rows, done = [], 0
    lock_errors = []

    def work(b):
        nonlocal done
        try:
            qs = ask_bundle(b)
        except Exception as e:
            import traceback
            lock_errors.append(traceback.format_exc(limit=2).strip().splitlines()[-1])
            return []
        done += 1
        if done % 10 == 0:
            print(f"  已处理 {done}/{len(bundles)} 束，累计 {len(rows)} 题", flush=True)
        # 把条号换成正文，便于后面查泄题
        body = {c["条号"]: c["正文"] for c in b}
        law = (b[0].get("简称") or b[0].get("law")) if b else ""
        for q in qs:
            nos = [str(n) for n in (q.get("条号") or [])]
            # 只保留确实在束里的条号，防模型编
            nos = [n for n in nos if n in body]
            if len(nos) < 2:
                continue
            rows.append({
                "问题": q.get("问题", "").strip(),
                "标准条号": nos,
                "标准法律": law,
                "每条的作用": q.get("每条的作用") or [],
            })
        return []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, bundles))

    # 去重（同一问题只留一条）
    seen, uniq = set(), []
    for r in rows:
        if r["问题"] and r["问题"] not in seen:
            seen.add(r["问题"])
            uniq.append(r)

    with open(DST, "w", encoding="utf-8") as f:
        for i, r in enumerate(uniq, 1):
            r["id"] = f"mh{i:03d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n处理 {done}/{len(bundles)} 束，产出 {len(uniq)} 道不重复的多跳题")
    if lock_errors:
        print(f"失败 {len(lock_errors)} 束，样例：{lock_errors[:3]}")
    print(f"已存 {DST}")
    print("下一步: python data/filter_multihop.py")


if __name__ == "__main__":
    main()
