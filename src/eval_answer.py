# -*- coding: utf-8 -*-
"""
评测（生成环节）：端到端"引用正确率"。简历上那两行数字就出自这里。

判定口径：
  一题算"引用正确" = 模型给出的**唯一**引用，指向的条文就是标准条号。
  多法语料下还要连法律名一起对（"第三条"在 4 部司法解释里都有，不校法名会假命中）。

两种模式，其余完全一致（同一 prompt、同一判分），这样对比才公平：
  --no-rag  不给任何资料，让模型凭记忆答
  --dense   把检索到的 top-5 喂给它再答

生成不会超过检索的天花板：dense 模式下 citations 只能来自捞到的那 5 条，
所以引用正确率上限 ≈ recall@5。09-22 在 n=100 上又验了一次，三组全部相等。

指标踩过的坑：最早判"标准条号在引用集合里就算对"，结果模型一次列 6 条法条撒网，
38 题全中 —— 指标太松、分数虚高，后面所有优化都看不出来。改成"只认唯一引用 +
精确匹配"之后才有意义。怀疑指标，先于怀疑结论。

用法:
  python src/eval_answer.py --no-rag                        # 民法典
  python src/eval_answer.py --dense --rewrite --retr hybrid
  python src/eval_answer.py laws --no-rag                   # 司法解释语料
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import re
import json
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (search, load_index, build_bm25, CHUNKS, VECTORS,
                      SFJS_CHUNKS, SFJS_VECTORS, NEW_CHUNKS, NEW_VECTORS)
import rewrite as rw

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_FILE = DATA_DIR / "评测结果.jsonl"
KEY_FILE = DATA_DIR / "api_key.txt"
TOP_K = 5
MAX_RETRY = 2
ASK_FAILURES = []   # API 调用失败的 (题号, 异常名)。最后统一判定 —— 见 ask()

# 语料配置：eval=评测集，idx=(chunks, vectors)
# 09-22 换 v2 评测集（每语料 100 题），v1 还在 data/ 里
CORPUS = {
    "minfadian": {"eval": DATA_DIR / "评测集_v2.jsonl", "idx": (CHUNKS, VECTORS)},
    "laws":      {"eval": DATA_DIR / "司法解释_评测集_v2.jsonl",
                  "idx": (SFJS_CHUNKS, SFJS_VECTORS)},
    "new":       {"eval": DATA_DIR / "新法_评测集_v2.jsonl",
                  "idx": (NEW_CHUNKS, NEW_VECTORS)},
}

# 多法语料下"法律名对没对"的判据：用该法最有辨识度的片段。
# 司法解释那几条要求"编名 + 解释"连在一起，免得把《民法典》总则编误判成《总则编解释》。
LAW_PATTERNS = [
    ("合同编通则", r"合同编通则"),
    ("总则编",     r"总则编.{0,10}解释"),
    ("婚姻家庭编", r"婚姻家庭编.{0,10}解释"),
    ("继承编",     r"继承编.{0,10}解释"),
    ("医疗保障法", r"医疗保障法"),
    ("生态环境法典", r"生态环境法典"),
]

_key = os.environ.get("DEEPSEEK_API_KEY") or (
    KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else None
)
if not _key:
    raise SystemExit("没找到 API key（环境变量 DEEPSEEK_API_KEY 或 data/api_key.txt）")
client = anthropic.Anthropic(base_url="https://api.deepseek.com/anthropic", api_key=_key)

# 条号归一化：模型可能写"第1254条"也可能写"第一千二百五十四条"，都折算成整数才能比。
# 这里原来复制了一份实现，跟 textutil 的那份会各自漂移 —— "第X条之一"的拦截就漏了它，
# 所以改成直接用同一份（sys.path 在上面已经加过 src/）。
from textutil import cn2int, ART_RE


def primary_citation(out):
    """取模型"最主要依据的那一条"，返回 (引用原文, 条号整数 or None)"""
    for x in out.get("citations") or []:
        t = str(x).strip()
        if not t:
            continue
        m = ART_RE.search(t)
        return t, (cn2int(m.group(0)) if m else None)
    # 回退：citations 空，就找正文里第一个"第X条"
    m = ART_RE.search(out.get("answer", ""))
    return (m.group(0) if m else ""), (cn2int(m.group(0)) if m else None)


def law_ok(cite_text, gold_law):
    """多法语料才校法律名；单一语料（民法典）直接放行"""
    for frag, pat in LAW_PATTERNS:
        if frag in gold_law:
            return re.search(pat, cite_text) is not None
    return True


ASK = """用户问了一个法律问题，请回答。

{citation_rule}
★ citations 里**只放一条**：你认为最关键、最直接的那一条法条引用。
  写法必须是"《法律名称》第X条"（例：《中华人民共和国民法典》第一千二百五十四条）。
  不要列一串备选。
只输出 JSON，不要任何解释：
{{"answer": "你的回答", "citations": ["《法律名称》第X条"]}}
{cotext}
问题：{q}
"""

NO_RAG_RULE = ("如果你在回答中引用了具体法条，就在 citations 里写你**最主要依据的那一条**，"
               "格式为《法律名称》第X条。**实在没有依据就给空数组**，不要硬凑。")


def build_prompt(q, ctx=None):
    if ctx:
        rule = ("**只能依据下面的资料回答**，资料不足以回答时就说无法确定。"
                "citations 必须来自资料里给出的条文，且**只放一条**。")
        block = "\n\n".join(f"【{c['条号']}｜出自《{c.get('简称') or '中华人民共和国民法典'}》】\n{c['正文']}"
                            for c in ctx)
        cotext = f"\n参考资料：\n{block}\n"
    else:
        rule = NO_RAG_RULE
        cotext = ""
    return ASK.format(citation_rule=rule, cotext=cotext, q=q)


def ask(prompt, qid=""):
    """失败必须被看见，不能静默降级成"引用错误"。

    09-22 之前这里是 `except Exception: return {"answer": "", "citations": []}`。
    后果很阴险：断网时每题都"正常"返回空引用 → 被判成引用错误 → 最后安安静静
    写一个 0% 的漂亮成绩进日志，跟真结果一模一样。宁可没有数字，也不要假数字。

    现在重试用尽后记一笔、返回 None，由 main 统一判定：偶发失败能容忍，
    系统性失败（断网/欠费）会被拦下、不写结果。
    """
    last = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            msg = client.messages.create(
                model="deepseek-v4-flash", max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            m = re.search(r"\{.*\}", text, flags=re.S)
            return json.loads(m.group(0))
        except Exception as e:
            last = e
            if attempt < MAX_RETRY:
                time.sleep(2)
    ASK_FAILURES.append((qid, type(last).__name__))
    return None


def main(mode, corpus="minfadian", use_rewrite=False, retr_mode="dense"):
    ASK_FAILURES.clear()          # 每次跑独立计数，别让上一轮语料的失败累进来
    conf = CORPUS[corpus]
    questions = [json.loads(l) for l in open(conf["eval"], encoding="utf-8")]
    chunks = vectors = bm25 = None
    if mode == "dense":
        chunks, vectors = load_index(*conf["idx"])
        bm25 = build_bm25(chunks)

    print(f"语料 {corpus}｜模式 {mode}｜检索 {retr_mode}"
          f"｜改写 {'开' if use_rewrite else '关'}｜{len(questions)} 题\n")
    details = []

    for q in questions:
        gold_no = cn2int(q["标准条号"][0])
        gold_law = q.get("标准法律", "")
        ctx = None
        if mode == "dense":
            query = rw.expand(q["问题"]) if use_rewrite else q["问题"]
            ctx = search(query, k=TOP_K, mode=retr_mode,
                         chunks=chunks, vectors=vectors, bm25=bm25)
        out = ask(build_prompt(q["问题"], [c for _, c in ctx] if ctx else None), qid=q["id"])
        if out is None:
            print(f"  ✗API {q['id']} 调用失败，跳过（不计入分母）")
            continue

        cite_text, got_no = primary_citation(out)
        hit_law = law_ok(cite_text, gold_law)
        hit_no = (got_no == gold_no)
        hit = hit_law and hit_no

        details.append({
            "id": q["id"], "类型": q["类型"], "命中": hit,
            "法律名对": hit_law, "条号对": hit_no,
            "标准": f'{gold_law[-20:]} 第{q["标准条号"][0]}',
            "模型引用": cite_text[:60],
        })
        flag = "✓" if hit else ("✗法" if not hit_law else "✗条")
        print(f"  {flag} {q['id']} [{q['类型']}] 正主 {gold_law[-14:]} {q['标准条号'][0]}"
              f"  ← 模型给 {cite_text[:34]}")
        time.sleep(0.4)

    # 失败率太高就拒绝写结果 —— 宁可没有数字，也不要一个假的 0%
    if ASK_FAILURES:
        print(f"\n★ API 调用失败 {len(ASK_FAILURES)} 题：{ASK_FAILURES[:5]}")
        if len(ASK_FAILURES) > 0.05 * len(questions):
            raise SystemExit(
                f"失败率 {len(ASK_FAILURES) / len(questions):.1%} > 5%，拒绝写结果。"
                "（为的是避免把网络故障写成一份 0% 的'成绩'）"
            )

    def rate(f, key="命中"):
        sub = [d for d in details if f(d)]
        return (sum(d[key] for d in sub) / len(sub) if sub else None), len(sub)

    r_all, n_all = rate(lambda d: True)
    r_ko, n_ko = rate(lambda d: d["类型"] == "口语化")
    r_yq, n_yq = rate(lambda d: d["类型"] == "法条原话")

    print("\n" + "=" * 60)
    print(f"引用正确率（n={n_all}）: {r_all:.1%}")
    if r_ko is not None:
        print(f"    口语化   (n={n_ko}): {r_ko:.1%}")
        print(f"    法条原话 (n={n_yq}): {r_yq:.1%}")
    print("=" * 60)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "tag": f"answer_{mode}_{retr_mode}{'_rw' if use_rewrite else ''}_{corpus}",
            "题数": n_all,
            "引用正确率": round(r_all, 4),
            "口语化": round(r_ko, 4) if r_ko is not None else None,
            "法条原话": round(r_yq, 4) if r_yq is not None else None,
        }, ensure_ascii=False) + "\n")
    print(f"已追加一行到 {LOG_FILE}（tag=answer_{mode}_{corpus}）")


if __name__ == "__main__":
    args = sys.argv[1:]
    corpus = next((a for a in args if a in CORPUS), "minfadian")
    mode = "dense" if "--dense" in args else "no-rag"
    retr_mode = "dense"
    if "--retr" in args:
        retr_mode = args[args.index("--retr") + 1]
    main(mode, corpus, use_rewrite="--rewrite" in args, retr_mode=retr_mode)
