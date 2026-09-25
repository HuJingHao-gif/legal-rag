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


def _validate(citations, hits):
    """引用校验：模型给的引用必须真的出现在喂给它的资料里。

    模型要是引了一条资料里没有的条文，说明它在凭记忆编 —— 这种引用对用户最危险，
    看着挺权威，但那条文可能早就修订或废止了。
    """
    allowed = {cn2int(c["条号"]) for _, c in hits}
    laws = {c.get("简称") or c.get("law", "") for _, c in hits}
    notes = []
    for cite in citations or []:
        m = ART_RE.search(str(cite))
        if not m:
            notes.append(f"引用格式无法解析：{cite}")
            continue
        no = cn2int(m.group(0))
        if no not in allowed:
            notes.append(f"引用 {cite} 不在检索到的资料里 —— 可能是模型凭记忆编的，不可信")
        elif not any((l and l in str(cite)) or ("民法典" in str(cite) and "民法典" in l) for l in laws):
            notes.append(f"引用 {cite} 条号在资料里，但法律名对不上，请人工核对")
    return (len(notes) == 0), notes


def ask(question, corpus="all", k=5, mode="hybrid", use_rewrite=True):
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
    prompt = PROMPT.format(ctx=_build_ctx(hits), q=question)
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
