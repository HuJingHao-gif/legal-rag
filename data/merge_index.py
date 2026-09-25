# -*- coding: utf-8 -*-
"""
把三套语料合成一个库，2802 条。

以前三套索引是分开的（民法典 / 司法解释 / 新法），页面上选哪个库就只在哪个库里搜，
问民法典的问题选了"新法"就永远搜不到。真实产品得一个库装所有法，
让检索自己决定引哪部。

坑一，行序契约：chunks 和 vectors 按行对齐，合并时两边顺序必须完全一致。
先按 SOURCES 的顺序拼 chunks，再按同一顺序 vstack 向量，错一个就全错位，而且不报错。
下面用同一个 SOURCES 列表驱动两边，最后再校一次行数。

坑二，条号跨法会撞："第一条"在民法典、4 部司法解释、2 部新法里都有。
所以每条必须带"法名简称"，评测时也得连法名一起比，否则假命中。
这里统一给每个 chunk 补上 `简称` 字段。

用法: python data/merge_index.py
输出: data/all_chunks.jsonl + data/all_vectors.npy
"""
import sys
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent

# (显示名, chunks 文件, vectors 文件, 缺省简称)
SOURCES = [
    ("民法典",   DATA_DIR / "民法典_chunks.jsonl",        DATA_DIR / "民法典_vectors.npy",       "民法典"),
    ("司法解释", DATA_DIR / "laws" / "司法解释_chunks.jsonl", DATA_DIR / "laws" / "司法解释_vectors.npy", None),
    ("新法",     DATA_DIR / "laws" / "新法_chunks.jsonl",     DATA_DIR / "laws" / "新法_vectors.npy",     None),
]

OUT_CHUNKS  = DATA_DIR / "all_chunks.jsonl"
OUT_VECTORS = DATA_DIR / "all_vectors.npy"


def main():
    all_chunks, all_vecs, report = [], [], []

    for name, cpath, vpath, default_short in SOURCES:      # 同一个列表驱动两边，顺序才对得上
        chunks = [json.loads(l) for l in open(cpath, encoding="utf-8")]
        vecs = np.load(vpath)
        if len(chunks) != vecs.shape[0]:
            raise SystemExit(f"[merge] {name} 自身就行序不一致：{len(chunks)} 条 vs {vecs.shape[0]} 行")

        for c in chunks:
            if default_short:
                c.setdefault("简称", default_short)         # 民法典原本没有简称，补上
            if not c.get("简称"):
                raise SystemExit(f"[merge] {name} 有条目缺简称：{c.get('条号')}")

        all_chunks += chunks
        all_vecs.append(vecs)
        report.append((name, len(chunks), vecs.shape[1]))

    vectors = np.vstack(all_vecs).astype("float32")

    # ── 校验 ──
    if len(all_chunks) != vectors.shape[0]:
        raise SystemExit(f"[merge] 行序契约被打破：{len(all_chunks)} 条 vs {vectors.shape[0]} 行")
    if len({v.shape[1] for v in all_vecs}) != 1:
        raise SystemExit("[merge] 各语料的向量维度不一致，不能直接 vstack")

    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(OUT_VECTORS, vectors)

    print(f"合并完成：{len(all_chunks)} 条 / {vectors.shape}")
    for name, n, dim in report:
        print(f"  {name:<8} {n:>5} 条  维度 {dim}")
    from collections import Counter
    print("\n按法分布（前 12）：")
    for k, v in Counter(c["简称"] for c in all_chunks).most_common(12):
        print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
