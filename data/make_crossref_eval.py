# -*- coding: utf-8 -*-
"""
交叉引用多跳题：一道题同时需要**一条司法解释**和**它引用的民法典条文**。

和 make_multihop_eval.py 的关键区别：**配对不是模型猜的，是法条里写死的**。
司法解释正文里明写着"民法典第X条"，所以"这两条要一起用"有文本依据，
经得起追问；而那批随机跨章配对的题，依据只是"模型觉得它俩有关系"。

代价是覆盖面窄 —— 只有 104 条司法解释显式引用了民法典（244 条里）。
顺带这批题是**跨法**的，正好能测合并库（真实产品形态），
而且用它的数字去验证那批随机配对题的结论，可以排除"出题方式"带来的偏差。

用法: python data/make_crossref_eval.py [输出.jsonl]
"""
import sys
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import llm

DATA_DIR = Path(__file__).parent
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "交叉引用_候选.jsonl"
WORKERS = 6
MAX_REF = 2          # 一条司法解释最多带几条民法典条文（多了就不是 2~3 跳了）

REF_PAT = re.compile(r"民法典第([一二三四五六七八九十百千零]+)条")

PROMPT = """下面给你一条中国**司法解释**，以及它正文里明确引用的**民法典**条文。

请设计一个**普通人真会问**的问题，回答它必须**同时**用到这几条。

硬要求：
1. **只靠司法解释那一条答不完整** —— 民法典那几条要提供它没有的信息
   （通常是上位规则、法律依据、或者构成要件）。
   如果司法解释自己就把话说全了、民法典那几条只是"顺带一提"，那这题**不合格**，返回空数组。
2. 用日常口语问，别照搬条文用语，不许出现"第X条"。
3. 对每一条说明它**独家**提供了什么（别的条给不了的）。

只输出 JSON 数组，不要解释：
[{{"问题": "...",
  "每条的作用": [{{"法": "司法解释简称", "条号": "第X条", "独家信息": "..."}},
                {{"法": "民法典", "条号": "第Y条", "独家信息": "..."}}]}}]

**凑不出合格的就返回 []**，宁缺毋滥。

待用条文：
{listing}
"""


def load():
    sj = [json.loads(l) for l in open(DATA_DIR / "laws" / "司法解释_chunks.jsonl", encoding="utf-8")]
    mf = [json.loads(l) for l in open(DATA_DIR / "民法典_chunks.jsonl", encoding="utf-8")]
    mf_by = {c["条号"]: c for c in mf}
    return sj, mf_by


def build_groups(sj, mf_by):
    """一条司法解释 + 它引用的民法典条文（最多 MAX_REF 条）"""
    groups = []
    for c in sj:
        refs = list(dict.fromkeys(f"第{m}条" for m in REF_PAT.findall(c["正文"])))
        arts = [mf_by[r] for r in refs if r in mf_by][:MAX_REF]
        if arts:
            groups.append((c, arts))
    return groups


def ask_group(sj_art, mf_arts):
    listing = f'【司法解释】《{sj_art["简称"]}》{sj_art["条号"]}\n{sj_art["正文"]}\n\n'
    listing += "\n\n".join(f'【民法典】{c["条号"]}\n{c["正文"]}' for c in mf_arts)
    out = llm.ask_json(PROMPT.format(listing=listing))
    return out if isinstance(out, list) else []


def main():
    sj, mf_by = load()
    groups = build_groups(sj, mf_by)
    print(f"司法解释 244 条 → {len(groups)} 组有显式引用的" )

    rows, done, errs = [], 0, []

    def work(g):
        nonlocal done
        sj_art, mf_arts = g
        try:
            qs = ask_group(sj_art, mf_arts)
        except Exception:
            errs.append(1)
            return
        done += 1
        if done % 20 == 0:
            print(f"  已处理 {done}/{len(groups)}，累计 {len(rows)} 题", flush=True)
        allowed = {(sj_art["简称"], sj_art["条号"])} | {("民法典", c["条号"]) for c in mf_arts}
        for q in qs:
            roles = q.get("每条的作用") or []
            gold = [(str(r.get("法")), str(r.get("条号"))) for r in roles]
            gold = [g2 for g2 in gold if g2 in allowed]
            if len(gold) < 2:
                continue
            rows.append({
                "问题": str(q.get("问题", "")).strip(),
                # ★ 金标跨两部法，所以每条都要带自己的法名（单跳评测集只有一个"标准法律"字段）
                "金标": [{"法": f, "条号": n} for f, n in gold],
                "每条的作用": roles,
            })

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, groups))

    seen, uniq = set(), []
    for r in rows:
        if r["问题"] and r["问题"] not in seen:
            seen.add(r["问题"])
            uniq.append(r)

    with open(DST, "w", encoding="utf-8") as f:
        for i, r in enumerate(uniq, 1):
            r["id"] = f"cr{i:03d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n处理 {done}/{len(groups)} 组，产出 {len(uniq)} 道不重复的交叉引用多跳题")
    if errs:
        print(f"失败 {len(errs)} 组")
    print(f"已存 {DST}")


if __name__ == "__main__":
    main()
