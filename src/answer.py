# -*- coding: utf-8 -*-
"""
核心业务逻辑：检索 + 生成 = 带引用的法律问答。
FastAPI 和页面都只是壳，这层才是资产。

一次请求：
    问题 ─(可选)查询改写─→ 检索 top-k ─→ 拼 prompt ─→ LLM ─→ 答案 + 引用
                                                            └→ 校验引用是不是真在资料里

命令行调试：
    python src/answer.py 高空抛物砸到人该找谁赔
    python src/answer.py --corpus new --no-rewrite 医保个人账户能不能给家人用
"""
import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (search, load_index, build_bm25,
                      CHUNKS, VECTORS, SFJS_CHUNKS, SFJS_VECTORS, NEW_CHUNKS, NEW_VECTORS,
                      ALL_CHUNKS, ALL_VECTORS)
from textutil import cn2int, ART_RE
import llm
import rewrite as rw

CORPORA = {
    # 默认 all：一个库装所有法，不用用户先想清楚"该在哪部法里搜"
    "all":       {"idx": (ALL_CHUNKS, ALL_VECTORS), "名称": "全部法律（民法典 + 司法解释 + 2026 新法）"},
    "minfadian": {"idx": (CHUNKS, VECTORS), "名称": "中华人民共和国民法典"},
    "laws":      {"idx": (SFJS_CHUNKS, SFJS_VECTORS), "名称": "司法解释"},
    "new":       {"idx": (NEW_CHUNKS, NEW_VECTORS), "名称": "2026 年新颁法律"},
}

_index_cache = {}      # corpus → (chunks, vectors, bm25)，不缓存的话每次请求都要重读重算


def _index(corpus):
    if corpus not in _index_cache:
        chunks, vectors = load_index(*CORPORA[corpus]["idx"])
        _index_cache[corpus] = (chunks, vectors, build_bm25(chunks))
    return _index_cache[corpus]


PROMPT = """你是法律助手。请**只依据下面提供的法条**回答用户的问题。

规则：
1. **只能依据资料回答**。资料不足以回答时，明确说"提供的资料不足以确定"，不要靠自己的记忆补充。
2. 回答要具体：说清结论、条件和依据。
3. `citations` 里**只放一条** —— 你最主要依据的那一条，写法为《法律名称》第X条。不要列一串备选。
4. 只输出 JSON，不要任何解释：
{{"answer": "你的回答", "citations": ["《法律名称》第X条"]}}

资料：
{ctx}

用户问题：{q}
"""


def _build_ctx(hits):
    """把检到的条文拼成给模型看的资料块"""
    blocks = []
    for _, c in hits:
        law = c.get("简称") or c.get("law") or "中华人民共和国民法典"
        blocks.append(f"【{c['条号']}｜出自《{law}》】\n{c['正文']}")
    return "\n\n".join(blocks)


PROMPT_MULTI = """你是法律助手。请**只依据下面提供的法条**回答用户的问题。

规则：
1. **只能依据资料回答**。资料不足以回答时，明确说"提供的资料不足以确定"，不要靠自己的记忆补充。
2. 回答要具体：说清结论、条件和依据。
3. `citations` 里列出你依据的**所有**条文，按重要性排序，最多 3 条。
   **只列真正用到的** —— 凑数的、沾边但没用上的都不要列。列多了等于没列。
4. 只输出 JSON，不要任何解释：
{{"answer": "你的回答", "citations": ["《法律名称》第X条"]}}

资料：
{ctx}

用户问题：{q}
"""

# ★ 为什么要有两套 prompt：单跳题（一问一条）用 PROMPT，强制唯一引用 ——
#   当初就是为了治"模型撒网列 6 条"才收紧的，单跳评测那批数字都是它跑出来的，
#   不能动。但多跳题需要 2~3 条才能答全，只让引一条就是结构上答不对。
#   所以按需切换，默认仍是老路径。


_LAW_RE = re.compile(r"《([^》]+)》")


def _validate(citations, hits):
    """★ 引用校验：模型给的引用必须真的出现在喂给它的资料里。

    这是"可核验"的落地——**如果模型引了一条不在资料里的法条，说明它在凭记忆编**，
    这种引用对用户是危险的（看着很权威，但可能已经修订/废止）。

    ★ 判据必须是 **(法名, 条号) 这一对**，不能分开判。合并库里"第一条"在 7 部法里都有，
    分开判会把"把生态环境法典第一条说成民法典第一条"这种错放过去 ——
    2026-09-27 写测试时撞出来的，当时的写法是 two 个独立的集合各自判断。
    """
    allowed = [((c.get("简称") or c.get("law") or ""), cn2int(c["条号"])) for _, c in hits]
    notes = []
    for cite in citations or []:
        s = str(cite)
        m_art = ART_RE.search(s)
        if not m_art:
            notes.append(f"引用格式无法解析：{cite}")
            continue
        no = cn2int(m_art.group(0))

        same_no = [law for law, n in allowed if n == no]
        if not same_no:
            notes.append(f"引用 {cite} 不在检索到的资料里 —— 可能是模型凭记忆编的，不可信")
            continue

        # 条号对上了，再看法名对不对得上（没写法名的没法查，放过）
        m_law = _LAW_RE.search(s)
        if m_law:
            cited_law = m_law.group(1)
            if not any(cited_law in l or l in cited_law for l in same_no):
                notes.append(f"引用 {cite} 条号在资料里，但法律名对不上，请人工核对")
    return (len(notes) == 0), notes


def ask(question, corpus="all", k=5, mode="hybrid", use_rewrite=True, multi_cite=False):
    """一次完整的问答，返回 dict，FastAPI 直接拿去序列化。

    mode 默认 hybrid 不挂 rerank。09-22 换 v2-m3 重测后，早期那个"rerank 反噬"
    已经没有了（早期那是 bge-reranker-base 太弱导致的），但收益落在噪声里，
    CPU 上还得 20 秒/题，不划算。真要挂就显式传 mode="hybrid+rerank"。
    """
    chunks, vectors, bm25 = _index(corpus)

    # ① （可选）查询改写：口语 → 法言法语 + 同义词扩充
    search_query = rw.expand(question) if use_rewrite else question

    # ② 检索
    hits = search(search_query, k=k, mode=mode, chunks=chunks, vectors=vectors, bm25=bm25)

    # ③ 生成
    prompt = (PROMPT_MULTI if multi_cite else PROMPT).format(ctx=_build_ctx(hits), q=question)
    try:
        out = llm.ask_json(prompt)
    except Exception as e:
        out = {"answer": f"生成失败：{e}", "citations": []}

    citations = out.get("citations") or []
    ok, notes = _validate(citations, hits)

    return {
        "question": question,
        "answer": out.get("answer", ""),
        "citations": citations,
        "引用校验": {"通过": ok, "说明": notes},
        "contexts": [{
            "条号": c["条号"],
            "法律": c.get("简称") or c.get("law", ""),
            "分数": round(float(s), 4),
            "正文": c["正文"],
        } for s, c in hits],
        "meta": {"语料": corpus, "模式": mode, "k": k, "改写": use_rewrite,
                 "检索用查询": search_query if use_rewrite else None},
    }


DIAGNOSE_PROMPT = """用户在问一个法律问题，下面是我已经检索到的法条。

问题：{q}

已检索到的条文：
{ctx}

请判断这些条文够不够**完整**回答这个问题。注意：问题可能同时涉及几件事，
只答上一部分不算够。

只输出 JSON，不要解释：
{{"够": true}}
或
{{"够": false, "还缺什么": "一句话说清缺哪部分", "新查询": "用来检索缺失内容的查询词"}}

「新查询」要写成像法条那样的法律术语，不要写条号。
"""


def _diagnose(question, hits):
    """让模型看已捞到的资料，判断够不够，不够就给出下一轮的查询。

    为什么不是简单的"够/不够"两分类：多跳题的失败模式不是"什么都没捞到"，
    而是"捞到第一跳、丢了第二跳"（实测 hit_any@5 有 81~92%，hit_all@5 只有 8~34%）。
    所以要的是**还缺什么、去哪找**，光知道"不够"没法决定下一步。
    """
    try:
        out = llm.ask_json(DIAGNOSE_PROMPT.format(q=question, ctx=_build_ctx(hits)))
    except Exception:
        return {"够": True, "新查询": None}     # 判定本身失败就当够了，别让循环空转烧 API
    return out if isinstance(out, dict) else {"够": True, "新查询": None}


def retrieve_agentic(question, corpus="all", k=5, mode="hybrid", use_rewrite=True,
                     max_rounds=3):
    """多轮检索：捞一轮 → 让模型看还缺什么 → 换个查询再捞一轮。

    返回 (rounds, queries)：
      rounds[i] = 到第 i 轮为止**累计去重**后的 [(分数, chunk), ...]
      queries   = 每轮实际用的查询

    只做检索、不生成答案 —— 评测要量的是"资料够不够全"，生成是另一回事。
    注意累计意味着上下文会变大，所以对比单轮时必须说清是在多大的上下文上比。
    """
    chunks, vectors, bm25 = _index(corpus)
    seen, collected, rounds, queries = set(), [], [], []

    query = rw.expand(question) if use_rewrite else question
    for r in range(max_rounds):
        queries.append(query)
        hits = search(query, k=k, mode=mode, chunks=chunks, vectors=vectors, bm25=bm25)
        for s, c in hits:
            key = (c.get("简称") or c.get("law"), c["条号"])
            if key not in seen:
                seen.add(key)
                collected.append((s, c))
        rounds.append(list(collected))

        if r == max_rounds - 1:
            break
        verdict = _diagnose(question, collected)
        if verdict.get("够"):
            break
        nxt = str(verdict.get("新查询") or "").strip()
        if not nxt:
            break
        query = nxt

    return rounds, queries


def ask_agentic(question, corpus="all", k=5, mode="hybrid", use_rewrite=True, max_rounds=3,
                multi_cite=False):
    """带多轮检索的问答。检索走 retrieve_agentic，生成和引用校验跟 ask() 完全一样。"""
    rounds, queries = retrieve_agentic(question, corpus, k, mode, use_rewrite, max_rounds)
    hits = rounds[-1]

    prompt = (PROMPT_MULTI if multi_cite else PROMPT).format(ctx=_build_ctx(hits), q=question)
    try:
        out = llm.ask_json(prompt)
    except Exception as e:
        out = {"answer": f"生成失败：{e}", "citations": []}

    citations = out.get("citations") or []
    ok, notes = _validate(citations, hits)

    return {
        "question": question,
        "answer": out.get("answer", ""),
        "citations": citations,
        "引用校验": {"通过": ok, "说明": notes},
        "contexts": [{
            "条号": c["条号"],
            "法律": c.get("简称") or c.get("law", ""),
            "分数": round(float(s), 4),
            "正文": c["正文"],
        } for s, c in hits],
        "meta": {"语料": corpus, "模式": mode, "k": k, "改写": use_rewrite,
                 "检索用查询": queries[0] if use_rewrite else None,
                 "检索轮数": len(rounds), "各轮查询": queries},
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    corpus = next((a for a in args if a in CORPORA), "all")
    # 把用掉的开关连同它的值一起摘掉，别漏进 query（踩过：--corpus 混进了问题里）
    args = [a for a in args if a not in CORPORA and a != "--corpus"]
    k = 5
    if "--k" in args:
        i = args.index("--k")
        k = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    use_rewrite = "--no-rewrite" not in args
    args = [a for a in args if a != "--no-rewrite"]
    q = " ".join(args) or "医保个人账户能不能给家人用？"

    r = ask(q, corpus=corpus, k=k, use_rewrite=use_rewrite)
    print(f"问题：{r['question']}")
    if r["meta"]["检索用查询"]:
        print(f"改写：{r['meta']['检索用查询'][:110]}")
    print(f"\n回答：{r['answer']}\n")
    print(f"引用：{r['citations']}   校验：{r['引用校验']}")
    print("\n依据：")
    for c in r["contexts"]:
        print(f"  {c['分数']:.4f}  {c['条号']}  《{c['法律']}》")
