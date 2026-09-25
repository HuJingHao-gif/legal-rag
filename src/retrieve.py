# -*- coding: utf-8 -*-
"""
检索：给一句话，返回 top-k 法条。评测和 FastAPI 都走这里。
2026-09 从 data/ 挪到 src/，因为 FastAPI 要 import 它。

踩的坑：vectors.npy 和 chunks.jsonl 只靠行序对齐，向量文件里根本没存条号。
重跑分块忘了重跑 embedding，分数照样出来、条文全错，还不报错。
load_index() 里加了行数校验，不一致直接退出。

四种模式（消融表每行都出自 search()）：
  dense     纯向量，换个说法也能找到；弱在专有名词和条号
  bm25      纯关键词，字面命中；弱在换个说法
  hybrid    两路 RRF 融合，两者错得不一样
  *_rerank  先粗排捞 pool 条，CrossEncoder 再精排

命令行调试：
  python src/retrieve.py 高空抛物砸到人该找谁赔
  python src/retrieve.py --mode hybrid 高空抛物砸到人该找谁赔
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像，得在 import 模型库之前
import sys
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

MODEL_NAME   = "BAAI/bge-small-zh-v1.5"
RERANK_NAME  = "BAAI/bge-reranker-v2-m3"
# 09-22 换的。之前用 bge-reranker-base（110M），改写之后再挂它反而全线变差，
# 当时判了"rerank 有害"；换成 v2-m3（568M）后反噬没了，是 base 太弱。
# 但 CPU 上 20 秒/题，收益又落在噪声里，所以产品默认还是不挂。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："   # bge 的查询侧前缀，文档侧不加

# 前三个单库，all 是三库合并 —— 页面默认走 all
CHUNKS       = DATA_DIR / "民法典_chunks.jsonl"
VECTORS      = DATA_DIR / "民法典_vectors.npy"
SFJS_CHUNKS  = DATA_DIR / "laws" / "司法解释_chunks.jsonl"
SFJS_VECTORS = DATA_DIR / "laws" / "司法解释_vectors.npy"
NEW_CHUNKS   = DATA_DIR / "laws" / "新法_chunks.jsonl"
NEW_VECTORS  = DATA_DIR / "laws" / "新法_vectors.npy"
ALL_CHUNKS   = DATA_DIR / "all_chunks.jsonl"
ALL_VECTORS  = DATA_DIR / "all_vectors.npy"

_embedder = None      # 懒加载，不然一 import 就等好几秒
_reranker = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_NAME)
    return _reranker


def load_index(chunks_path=None, vectors_path=None):
    """读 chunk 和向量，顺手校一下行序对齐没对齐。"""
    chunks_path  = Path(chunks_path)  if chunks_path  else CHUNKS
    vectors_path = Path(vectors_path) if vectors_path else VECTORS
    chunks  = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    vectors = np.load(vectors_path)
    if len(chunks) != vectors.shape[0]:
        raise SystemExit(
            f"行序契约被打破：{chunks_path.name} {len(chunks)} 条，"
            f"但 {vectors_path.name} {vectors.shape[0]} 行。\n"
            "重跑分块脚本后必须成对重跑 embedding 脚本。"
        )
    return chunks, vectors


def build_bm25(chunks):
    """建 BM25 索引。第一次约 1 秒，之后复用就行，别反复建。"""
    from bm25 import BM25
    return BM25([c["检索文本"] for c in chunks])


def _rank_of(order):
    """把"按分数降序的下标数组"转成 {下标: 名次}"""
    return {int(i): r for r, i in enumerate(order, 1)}


def search(query, k=10, mode="dense", chunks=None, vectors=None, bm25=None,
           pool=50):
    """统一检索入口。

    mode: dense | bm25 | hybrid | dense+rerank | bm25+rerank | hybrid+rerank
    pool: 挂 rerank 时粗排捞多少条候选给它精排
    返回 [(分数, chunk), ...]。分数只在同一个 mode 内部有意义，跨 mode 比大小没意义。
    """
    if chunks is None or vectors is None:
        chunks, vectors = load_index()
    use_rerank = mode.endswith("+rerank")
    base = mode.replace("+rerank", "")

    # 粗排
    if base == "dense":
        qv = _get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = vectors @ qv                      # 归一化之后点积就是余弦
        cand = np.argsort(-scores)[: pool if use_rerank else k]
        ranked = [(float(scores[i]), int(i)) for i in cand]

    elif base == "bm25":
        bm25 = bm25 or build_bm25(chunks)
        scores = bm25.scores(query)
        cand = np.argsort(-scores)[: pool if use_rerank else k]
        ranked = [(float(scores[i]), int(i)) for i in cand]

    elif base == "hybrid":
        from bm25 import rrf_fuse
        bm25 = bm25 or build_bm25(chunks)
        ds = vectors @ _get_embedder().encode([QUERY_PREFIX + query],
                                              normalize_embeddings=True)[0]
        bs = bm25.scores(query)
        n_each = pool if use_rerank else max(k, 20)
        d_ranks = _rank_of(np.argsort(-ds)[:n_each])
        b_ranks = _rank_of(np.argsort(-bs)[:n_each])
        fused = rrf_fuse([d_ranks, b_ranks])[: pool if use_rerank else k]
        ranked = [(float(s), int(i)) for i, s in fused]

    else:
        raise SystemExit(f"未知 mode：{mode}")

    # 精排（可选）
    if use_rerank and ranked:
        rr = _get_reranker()
        pairs = [[query, chunks[i]["检索文本"]] for _, i in ranked]
        rs = rr.predict(pairs)
        order = np.argsort(-np.asarray(rs))
        ranked = [(float(rs[o]), ranked[o][1]) for o in order[:k]]

    return [(s, chunks[i]) for s, i in ranked[:k]]


def retrieve(query, k=10, chunks=None, vectors=None):
    """纯向量检索。老接口，留着给早期脚本用。"""
    return search(query, k=k, mode="dense", chunks=chunks, vectors=vectors)


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "dense"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        args = args[:i] + args[i + 2:]
    query = " ".join(args) or "高空抛物致人损害由谁承担责任"
    print(f"查询：{query}\n模式：{mode}\n")
    for rank, (score, c) in enumerate(search(query, k=5, mode=mode), 1):
        loc = c.get("简称") or c.get("编") or ""
        print(f"{rank}. {score:.4f}  {c['条号']}  （{loc}）")
        print(f"    {c['正文'][:80]}…")
