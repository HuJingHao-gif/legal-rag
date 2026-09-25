# -*- coding: utf-8 -*-
"""
BM25：稀疏检索，核心逻辑约 30 行，除了 jieba 没别的依赖。

为什么要有它：dense 把文本压成 512 个数字，擅长"大概在说什么"，不擅长字面精确匹配。
用户问"高空抛物"、法条写"从建筑物抛掷物品"，一个字都不重合，向量反而能对上；
但问"第一千二百五十四条"，或者撞上一个法条里的专有名词，向量就糊了。
BM25 正好相反 —— 只看字面重合，专有名词一击必中，换个说法就完全找不到。
两者错得不一样，所以 hybrid 比任何一路单独都强。

公式：score(q,d) = Σ_t IDF(t) · f(t,d)(k1+1) / [ f(t,d) + k1(1 − b + b·|d|/avgdl) ]

三个设计点：
  IDF        ln(1 + (N − n + 0.5)/(n + 0.5))。稀有词区分度高 —— "的"到处都有 IDF≈0，
            等于没用；"高空抛物"只在 3 条里出现过，一命中就是强信号。
  词频饱和    k1=1.5。一个词出现 10 次不该比出现 2 次重要 5 倍，f 同时在分子和分母里，
            所以收益递减。TF-IDF 是线性的，会过誉长文档。
  长度归一    b=0.75。长条文天然更容易命中查询词（字多），得打个折。
            b=1 完全归一，b=0 不归一，0.75 是经验值。
"""
import re
from collections import Counter

from math import log

import jieba
import numpy as np

_PUNCT = re.compile(r"[\s，。？、；：（）《》【】()\[\]{}"
                    r"""。！？；：""''\.\,\?\!\:\;\"'—－\-/]+""")


def tokenize(text):
    """中文分词 → 去标点空白 → 小写。中文没天然空格，检索必须先分词。"""
    words = jieba.lcut(text)
    return [w.lower() for w in words if w.strip() and not _PUNCT.fullmatch(w)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tokens = [tokenize(d) for d in docs]
        self.lens = np.array([len(t) for t in self.tokens], dtype="float32")
        self.avgdl = float(self.lens.mean()) or 1.0
        self.tf = [Counter(t) for t in self.tokens]

        # 文档频率 df：多少篇文档含这个词。得用 set 去重，不是数总出现次数
        df = Counter()
        for t in self.tokens:
            df.update(set(t))
        # 加 1 平滑的 IDF，保证恒正，免得常见词给出负分
        self.idf = {w: log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):
        """给全部文档打分，返回 shape=(N,)。分数无上界，和余弦不是一个量纲，别混着比。"""
        out = np.zeros(self.N, dtype="float32")
        for w in set(tokenize(query)):           # set：同一个词在查询里出现多次不重复累加
            idf = self.idf.get(w)
            if not idf:
                continue                          # 这词不在任何文档里，跳过
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def rrf_fuse(rank_lists, k=60):
    """Reciprocal Rank Fusion：按名次融合多路排名，不按分数。

    不能直接相加分数：dense 是余弦（0~1），BM25 无上界（30 也可能是 300），
    量纲完全不同，直接加权等于让 BM25 单方面说了算。RRF 只用"排第几"，
    天然免疫量纲问题，这也是它成了混合检索默认做法的原因。

    score(d) = Σ_routes 1 / (k + rank)，k=60 是原论文的经验值。
    排第 1 得 1/61、第 2 得 1/62 …… 名次越前贡献越大，且收益递减。
    """
    fused = {}
    for ranks in rank_lists:                  # 每路是一个 {doc_idx: rank}
        for idx, rank in ranks.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
