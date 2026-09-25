# -*- coding: utf-8 -*-
"""
查询改写：把"普通人怎么问"改写成"法条怎么写"。检索前加一个 LLM 步骤。

为什么需要 —— 一个真实失败案例：
  用户问「我让别人帮我办事，他不好好办或者只办一半，把我害了，他该不该赔我？」
  正主是《民法典》第一百六十四条（代理人不履行职责造成被代理人损害的责任），
  正文是「代理人不履行或者不完全履行职责，造成被代理人损害的，应当承担民事责任…」
  **问题和法条一个字都不重合。** 向量能对上"大概在说办事"，
  但"代理 / 被代理人"这层法律概念没被激活 —— recall@1 只有 57.7%，根子就在这。

  改写成「委托代理 代理人不履行职责 不完全履行职责 造成被代理人损害 民事责任 …」
  术语一到位，dense 和 bm25 都能命中。

也是 agentic 最朴素的形态：检索前加一个 LLM 步骤。不新潮，但是被数据逼出来的 ——
光是口语化题就丢 40 多个百分点，hybrid / rerank 都治不了，因为它们都在"排序"层面，
而这里的问题是"查询词本身就没带对信号"。

用法:
  python src/rewrite.py "我让别人帮我办事，他不好好办把我害了"
  python src/rewrite.py --cache-stats
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm import ask

CACHE_FILE = Path(__file__).parent.parent / "data" / "改写缓存.jsonl"

PROMPT = """用户在问一个法律问题（口语）。请把它改写成**适合在法条库里做检索的查询**。

要求：
1. 把口语说法换成法律术语（例："让别人帮我办事"→"委托代理"；"砸到人"→"造成他人损害"；"把我关起来不让走"→"非法限制人身自由"）
2. 补出相关的法律概念和同义词，用空格隔开，方便关键词检索
3. **必须保留原问题的事实限定**（否则会检索到不相关的条文）
4. ★ **绝对不要输出任何条号**（不许出现"第X条"或具体数字）。
   你的任务是**扩充术语**，不是回忆答案 —— 把答案写进查询里会让评测失真。
5. 只输出改写后的查询文本，一行，不要解释、不要引号

原问题：{q}
"""

# 兜底：万一模型还是写了条号（要求 4 那道理没拦住），这里机械地再剥一遍
import re as _re
_ARTNO = _re.compile(r"(《[^》]*》)?第[一二三四五六七八九十百千零\d]+条(之[一二三四五六七八九十]+)?")


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    return {json.loads(l)["原文"]: json.loads(l)["改写"]
            for l in CACHE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def rewrite(q, cache=None, verbose=False):
    """返回改写后的查询。带磁盘缓存，同一句问题只花一次 API 调用。"""
    cache = cache if cache is not None else _load_cache()
    if q in cache:
        return cache[q]
    out = ask(PROMPT.format(q=q)).strip().strip('"').strip("“”").replace("\n", " ")
    out = _ARTNO.sub(" ", out)          # 剥条号，防"把答案写进查询里"
    out = _re.sub(r"\s+", " ", out).strip()
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"原文": q, "改写": out}, ensure_ascii=False) + "\n")
    if verbose:
        print(f"  改写: {out[:100]}")
    return out


def expand(q, cache=None):
    """检索实际用的查询 = 原文 + 改写。

    两个都留的原因：改写可能丢掉事实限定（"阳台上花盆掉下来"被改写成泛的
    "物件损害责任"），原文保留具体事实，两路信号互补。
    """
    return f"{q} {rewrite(q, cache)}"


def prewarm(eval_set):
    """把一个评测集里的问题全改写一遍写进缓存，之后跑消融就不再花 API 了。"""
    qs = [json.loads(l) for l in open(eval_set, encoding="utf-8")]
    cache = _load_cache()
    todo = [q["问题"] for q in qs if q["问题"] not in cache]
    print(f"{eval_set}：{len(qs)} 题，需新改写 {len(todo)} 条（已缓存 {len(qs)-len(todo)}）")
    for i, q in enumerate(todo, 1):
        r = rewrite(q, cache)
        print(f"  [{i}/{len(todo)}] {r[:78]}")
    print("完成")


def cache_stats():
    c = _load_cache()
    print(f"缓存条数: {len(c)}  → {CACHE_FILE}")
    for i, (k, v) in enumerate(c.items()):
        if i >= 5:
            break
        print(f"  原文: {k[:40]}")
        print(f"  改写: {v[:90]}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--cache-stats" in args:
        cache_stats()
    elif "--prewarm" in args:
        prewarm(args[args.index("--prewarm") + 1])
    elif args:
        q = " ".join(args)
        print(f"原文: {q}")
        print(f"改写: {rewrite(q, verbose=False)}")
    else:
        print(__doc__)
