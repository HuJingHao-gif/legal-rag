# -*- coding: utf-8 -*-
"""
embedding：把 1260 条 chunk 的"检索文本"编码成向量，存 npy。
模型 BAAI/bge-small-zh-v1.5，中文检索够用、512 维、本地能跑。
向量文件里压根没存条号，chunks.jsonl 和 vectors.npy 纯靠行序对齐——
以后重跑分块，务必连 embedding 一起重跑，不然分数照样出、条文全错，还不报错。
输入  data/民法典_chunks.jsonl
输出  data/民法典_vectors.npy（1260 × 512，行序对上 jsonl）
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像，必须在 import sentence_transformers 之前
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent
src = DATA_DIR / "民法典_chunks.jsonl"
dst = DATA_DIR / "民法典_vectors.npy"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 读 chunk。行序就是向量行序，后面靠它把向量对回条文
chunks = [json.loads(l) for l in open(src, encoding="utf-8")]
texts = [c["检索文本"] for c in chunks]
print(f"待编码 chunk 数: {len(texts)}")

# 加载模型。首次会自动下 ~100MB 到本地缓存，之后就快
print("加载模型…（首次会下载，稍等）")
model = SentenceTransformer(MODEL_NAME)

# normalize_embeddings=True：归一化之后，点积就等于余弦相似度
vectors = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)
vectors = np.asarray(vectors, dtype="float32")
np.save(dst, vectors)
print(f"向量形状: {vectors.shape}  已存: {dst}")

# 自测：拿一句话查一查，看向量是不是真懂语义
# bge 中文模型查询侧要加这句指令前缀（文档侧不加），召回会明显好
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
query = QUERY_PREFIX + "高空抛物致人损害由谁承担责任"
qv = model.encode([query], normalize_embeddings=True)[0]

scores = vectors @ qv                       # 矩阵乘：1260×512 · 512 → 1260 个相似度
top = np.argsort(-scores)[:5]               # 分数最高的 5 个
print("\n自测检索：高空抛物致人损害由谁承担责任")
for idx in top:
    c = chunks[idx]
    print(f"  {scores[idx]:.3f}  {c['条号']}  {c['正文'][:100]}")
