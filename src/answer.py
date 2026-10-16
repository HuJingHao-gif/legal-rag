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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (search, load_index, build_bm25, warm_embedder,
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


def warmup(corpus="all"):
    """启动预热：把默认语料的索引（含 BM25）和 embedding 模型先加载好。

    不预热的话，冷启动后的第一个问题要等 47~66 秒 —— 加载 bge-small，再加上
    给 2802 条建 BM25。这段时间要么第一个用户等，要么服务器启动时等。选后者：
    启动慢是确定的、看得见的，用户等是随机的、看起来像卡死。

    只热默认语料。切到别的语料时那一份还是得现建，但那是用户主动切的，
    等一次能接受；四份全建会把启动时间翻几倍。

    失败就让它抛 —— 索引文件不在、或和向量行数对不上，早死比晚死好。
    """
    t0 = time.time()
    _index(corpus)
    warm_embedder()
    print(f"预热完成：{corpus} 索引 + embedding 模型，用时 {time.time() - t0:.1f}s")


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
# 正文里把"法名"和"条号"一起扫。法名是用来给**后面**出现的条号定语境的 ——
# 法条行文就是这么走的（"《医疗保障法》第三十二条规定…"），所以见到法名就更新当前语境。
_PROSE_RE = re.compile(r"《([^》]+)》|(" + ART_RE.pattern + ")")


def _judge(no, law, allowed, traceable=frozenset()):
    """判一条引用过不过，没问题返回 None。

    ★ 判据必须是 **(法名, 条号) 这一对**，不能分开判。合并库里"第一条"在 7 部法里都有，
    分开判会把"把生态环境法典第一条说成民法典第一条"这种错放过去 ——
    2026-09-27 写测试时撞出来的，当时的写法是两个独立的集合各自判断。

    traceable = "模型看到的材料正文里出现过的条号"。命中它不算编造 ——
    模型是在复述资料原文，不是自己加的。
    """
    same_no = [l for l, n in allowed if n == no]
    if not same_no:
        if no in traceable:
            return None
        return "不在检索到的资料里 —— 可能是模型凭记忆编的，不可信"
    if law and not any(law in l or l in law for l in same_no):
        return "条号在资料里，但法律名对不上，请人工核对"
    return None


def _validate(citations, hits, answer="", question=""):
    """★ 引用校验：模型给的引用必须真的出现在喂给它的资料里。

    这是"可核验"的落地——**如果模型引了一条不在资料里的法条，说明它在凭记忆编**，
    这种引用对用户是危险的（看着很权威，但可能已经修订/废止）。

    ★ 要查**两处**，只查 citations 是不够的：
      1. `citations` 字段 —— 模型"正式声明"的引用
      2. **回答正文里点名的条号** —— 2026-10-10 补的。k=5 时正文可能顺手带出
         四五条法条，而 citations 规定只有一条；只查 citations 的话，页面上那句
         "引用可核验"就只保住了其中一条，其余全在裸奔。
         触发点是用户在页面上撞出来的：同一道题两次回答，一次点名 4 条、
         一次只点名 1 条 —— 那 4 条当时**没人查**。

    ★ question 是干什么的：模型复述用户提问时，会把提问里的条号带进正文。
    那种照样要提示"无法核验"（它确实不在资料里），但不该说成"模型凭记忆编的" ——
    条号是用户自己给的。不区分的话，凡是"提问里带条号"的题都会冤枉模型。
    """
    allowed = [((c.get("简称") or c.get("law") or ""), cn2int(c["条号"])) for _, c in hits]
    notes = []

    for cite in citations or []:
        s = str(cite)
        m_art = ART_RE.search(s)
        if not m_art:
            notes.append(f"引用格式无法解析：{cite}")
            continue
        m_law = _LAW_RE.search(s)
        problem = _judge(cn2int(m_art.group(0)), m_law.group(1) if m_law else None, allowed)
        if problem:
            notes.append(f"引用 {cite} {problem}")

    # 问题里出现过的条号。用 ART_RE 直接扫、不走 articles_in ——
    # 后者碰到「第X条之一」会抛错，而用户的问题里什么写法都可能有，不该把请求打挂。
    asked = {cn2int(m.group(0)) for m in ART_RE.finditer(question or "")}

    # ★ 模型"看到过"的条号：检索到的条文**正文里**提到的那些。
    # 司法解释和民法典的条文常把外部法条写进自己正文（合同编通则解释第二十八条结尾就写着
    # "依据……《担保制度解释》第六十八条的规定处理"），模型照抄资料时会把那个条号一并带出来。
    # 不排除掉的话这类题会稳定误报：12 题抽样里两次告警，**两次都是这个原因**，
    # 而且第一次我还把它误判成了"真命中" —— 只查了"语料里有没有这部法"，
    # 没去查检索到的正文里写没写这个条号。判据的误报也得靠查，不能靠推。
    in_text = {cn2int(m.group(0)) for _, c in hits for m in ART_RE.finditer(c.get("正文") or "")}
    retrieved = {n for _, n in allowed}

    # 正文：同一个条号出现多次只报一次
    law, seen = None, set()
    for m in _PROSE_RE.finditer(answer or ""):
        if m.group(1):
            law = m.group(1)
            continue
        tok, no = m.group(2), cn2int(m.group(2))
        if (law, no) in seen:
            continue
        seen.add((law, no))
        where = f"{tok}（《{law}》）" if law else tok

        if no not in retrieved and no not in in_text and no in asked:
            # 出处是提问本身，不是资料。照样说一声"无法核验"，但这是复述、不是编造。
            notes.append(f"正文提到 {where} —— 它在你问的问题里出现过，但不在检索到的资料里，无法核验")
            continue

        problem = _judge(no, law, allowed, in_text)
        if problem:
            notes.append(f"正文提到 {where} {problem}")

    return (len(notes) == 0), notes


_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')
_JSON_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
             '"': '"', "\\": "\\", "/": "/"}


def _partial_answer(buf):
    """从**还没吐完**的 JSON 里，抠出 answer 字段到目前为止收到的那部分。

    为什么不干脆让模型吐纯文本：PROMPT 的输出格式是评测口径的一部分，
    citations 就藏在这段 JSON 里，改了格式单跳那批历史数字就对不上了。
    所以格式不动，把 JSON 流在服务端切开，只把 answer 的字符串值推给页面。

    每来一小段就整段重扫一遍。输出统共几 KB，重算的成本可以忽略，
    换来的是不用维护"转义符只收到一半"那种状态机 —— 那种东西就算写错了
    也只是少一个字，不报错，正是这个项目最怕的静默失败。
    """
    m = _ANSWER_KEY.search(buf)
    if not m:
        return None
    i, out = m.end(), []
    while i < len(buf):
        ch = buf[i]
        if ch == "\\":
            nxt = buf[i + 1:i + 2]
            if not nxt:
                break                       # 转义符刚吐了一半，等下一段
            if nxt == "u":
                hexs = buf[i + 2:i + 6]
                if len(hexs) < 4:
                    break
                try:
                    out.append(chr(int(hexs, 16)))
                except ValueError:
                    break                   # \u 后面收到的还不是十六进制，等下一段
                i += 6
            else:
                out.append(_JSON_ESC.get(nxt, nxt))
                i += 2
            continue
        if ch == '"':
            break                           # 引号闭合，answer 收完了
        out.append(ch)
        i += 1
    return "".join(out)


def _contexts(hits):
    """检索结果 → 给页面/接口看的依据列表"""
    return [{
        "条号": c["条号"],
        "法律": c.get("简称") or c.get("law", ""),
        "分数": round(float(s), 4),
        "正文": c["正文"],
    } for s, c in hits]


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
    ok, notes = _validate(citations, hits, out.get("answer", ""), question)

    return {
        "question": question,
        "answer": out.get("answer", ""),
        "citations": citations,
        "引用校验": {"通过": ok, "说明": notes},
        "contexts": _contexts(hits),
        "meta": {"语料": corpus, "模式": mode, "k": k, "改写": use_rewrite,
                 "检索用查询": search_query if use_rewrite else None},
    }


def ask_stream(question, corpus="all", k=5, mode="hybrid", use_rewrite=True,
               multi_cite=False):
    """流式版 ask()：检索和引用校验照旧，只有"生成"这一步变成边收边吐。

    yield (事件名, 数据) 二元组，app.py 包成 SSE。事件顺序：
        meta           检索依据（条号/分数/正文）
        delta *        正文前缀 —— 每次都发**全量**，页面直接覆盖，不做累加
        done           最终答案 + 引用 + 校验结论

    先推 meta 不是随手排的：检索本来就得先跑完（hybrid 在 CPU 上要几秒），
    先把依据推下去，页面在等模型第一个字的时候就有东西可看。

    生成中途出错就让它抛 —— 抛出去页面会显示"生成中断"。半截 JSON 意味着
    答案本来就残缺，这时候安静地拿已经收到的半截当答案，等于把截断的正文
    冒充成成品（这个项目在假成绩上栽过四次，不能再添一次）。
    """
    chunks, vectors, bm25 = _index(corpus)
    search_query = rw.expand(question) if use_rewrite else question
    hits = search(search_query, k=k, mode=mode, chunks=chunks, vectors=vectors, bm25=bm25)

    yield "meta", {
        "question": question,
        "contexts": _contexts(hits),
        "meta": {"语料": corpus, "模式": mode, "k": k, "改写": use_rewrite,
                 "检索用查询": search_query if use_rewrite else None},
    }

    prompt = (PROMPT_MULTI if multi_cite else PROMPT).format(ctx=_build_ctx(hits), q=question)
    buf, shown = "", ""
    for piece in llm.ask_stream(prompt):
        buf += piece
        got = _partial_answer(buf)
        if got is not None and got != shown:
            shown = got
            yield "delta", {"text": shown}

    out = llm.parse_json(buf)
    citations = out.get("citations") or []
    ok, notes = _validate(citations, hits, out.get("answer", ""), question)
    yield "done", {
        "answer": out.get("answer", ""),
        "citations": citations,
        "引用校验": {"通过": ok, "说明": notes},
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
    ok, notes = _validate(citations, hits, out.get("answer", ""), question)

    return {
        "question": question,
        "answer": out.get("answer", ""),
        "citations": citations,
        "引用校验": {"通过": ok, "说明": notes},
        "contexts": _contexts(hits),
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
