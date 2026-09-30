# -*- coding: utf-8 -*-
"""
合并索引闸门的测试：行序契约 + 简称。

为什么单独测它：chunks.jsonl 和 vectors.npy 之间没有键，只有**行序**。
合并时只要两边顺序错一格，第 3 条的正文就会配上第 7 条的向量 ——
检索照样返回结果、分数照样正常，只是拿出来的条文全是错的。
这是项目里最典型的"静默错误"，所以它值得一个测试盯着"错了有没有大声报出来"。

测法：把 SOURCES 和输出路径换成临时目录，跑真实的 main()，
然后看它到底抛没抛、消息里说的是不是那一件事。

运行: python tests/test_merge_index.py
"""
import io
import json
import sys
import tempfile
import contextlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import merge_index


def write_corpus(d, name, chunks, dim=4, vec_rows=None):
    """写一套语料。vec_rows 用来故意造出"条数和向量行数不一致"的坏数据"""
    cpath = d / f"{name}_chunks.jsonl"
    vpath = d / f"{name}_vectors.npy"
    with open(cpath, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    rows = len(chunks) if vec_rows is None else vec_rows
    np.save(vpath, np.zeros((rows, dim), dtype="float32"))
    return cpath, vpath


def run_merge(sources, d):
    """把 main() 的输入输出都指到临时目录，跑一遍。

    返回 (异常描述 或 None, 输出的 chunks 行数 或 None, 输出的向量行数 或 None)。
    SystemExit 不继承 Exception，所以要接 BaseException 才能连"大声退出"一起抓住。
    """
    merge_index.SOURCES = sources
    merge_index.OUT_CHUNKS = d / "all_chunks.jsonl"
    merge_index.OUT_VECTORS = d / "all_vectors.npy"

    err = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            merge_index.main()
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"

    if err is not None:
        return err, None, None
    n_chunks = len(open(merge_index.OUT_CHUNKS, encoding="utf-8").readlines())
    n_vecs = np.load(merge_index.OUT_VECTORS).shape[0]
    return None, n_chunks, n_vecs


def main():
    ok = bad = 0

    def check(desc, got, want):
        nonlocal ok, bad
        if got == want:
            ok += 1
            print(f"  ✓ {desc}")
        else:
            bad += 1
            print(f"  ✗ {desc}")
            print(f"      期望 {want}，实际 {got}")

    def check_msg(desc, got, keyword):
        # 只查异常消息里有没有那个关键词：措辞以后可以改，要保住的是「报了哪件事」
        check(desc, bool(got) and keyword in got, True)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        print("正常合并")
        # 法甲故意不带 简称，靠 SOURCES 里的默认简称补（民法典就是这种情况）
        a = write_corpus(d, "法甲", [{"条号": "第一条"}, {"条号": "第二条"}])
        # 法乙自带 简称（司法解释/新法就是这种情况）
        b = write_corpus(d, "法乙", [{"条号": "第一条", "简称": "法乙"}])
        err, n_chunks, n_vecs = run_merge(
            [("法甲", a[0], a[1], "法甲"), ("法乙", b[0], b[1], None)], d)
        check("两套语料都能合并，不抛异常", err, None)
        check("输出向量行数 == 输出条数（行序契约成立）", [n_chunks, n_vecs], [3, 3])

        out = [json.loads(l) for l in open(merge_index.OUT_CHUNKS, encoding="utf-8")]
        check("每一路都补上了默认简称",
              [c.get("简称") for c in out], ["法甲", "法甲", "法乙"])
        check("按 SOURCES 的顺序拼，法甲在前法乙在后",
              [c["简称"] for c in out].index("法乙"), 2)

        print("\n坏数据必须大声报错，不许静默产出错位的索引")
        # 单套语料自身就不对齐：条数和向量行数不等
        c = write_corpus(d, "法丙", [{"条号": "第一条"}, {"条号": "第二条"}],
                         vec_rows=3)
        err, _, _ = run_merge([("法丙", c[0], c[1], "法丙")], d)
        check("抛的是 SystemExit（不是静默继续）",
              err.split(":")[0] if err else None, "SystemExit")
        check_msg("消息点明是这套语料自身行序不一致", err, "自身就行序不一致")

        # 既没有 简称，SOURCES 也没给默认简称 —— 这条数据没法判断属于哪部法
        e = write_corpus(d, "法丁", [{"条号": "第一条"}])
        err, _, _ = run_merge([("法丁", e[0], e[1], None)], d)
        check("缺简称也抛 SystemExit",
              err.split(":")[0] if err else None, "SystemExit")
        check_msg("消息里带上了出问题的条号", err, "缺简称")

        # 已知：代码里有"向量维度不一致"的闸门，但它走不到 ——
        # np.vstack 在维度不同的时候自己先抛 ValueError，所以那条 SystemExit 消息
        # 是死代码。不影响"失败必须大声"（确实抛了），只是报错来源变成了 numpy，
        # 消息里没有中文说明。要留住的是"它就是会抛"。
        f = write_corpus(d, "法戊", [{"条号": "第一条"}], dim=4)
        g = write_corpus(d, "法己", [{"条号": "第一条", "简称": "法己"}], dim=5)
        err, _, _ = run_merge(
            [("法戊", f[0], f[1], "法戊"), ("法己", g[0], g[1], None)], d)
        check("向量维度不一致会抛（由 numpy 抛出，不是那条中文闸门）",
              err.split(":")[0] if err else None, "ValueError")

    print(f"\n{ok}/{ok + bad} 通过")
    if bad:
        print(f"有 {bad} 个用例失败 —— 合并索引的闸门失效了，行序错位会静默发生")
        return 1
    print("合并闸门有效：坏数据会大声退出，好数据的行序契约成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
