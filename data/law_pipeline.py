# -*- coding: utf-8 -*-
"""
通用法律管道：抓取 → 清洗 → 分块 → 向量。多部法共用一套代码，差异全在 SETS 配置里。

SETS 现在两个：
  sfjs  4 部司法解释，244 条
  new   2026 年新颁法律（模型知识截止之后），决定性实验靠它

没有直接复用民法典那套（clean_reflow.py + chunk.py）：
  民法典出来的格式是条号独立成行、正文在下一行，
  司法解释和新法的 OFD 里条号和正文粘在一行（"第三条借贷双方就…"）。
  这套改成按"行首是不是 第X条 开头"切条，两种格式都能吃。

清洗规则是按 OFD 渲染瑕疵写的（页脚数字、断行、标题区），几部法同源，能覆盖大部分情况。
每个阶段跑完先校验，不合格直接退出，别让它沉默出垃圾。

用法:
  python data/law_pipeline.py all sfjs      # 司法解释集
  python data/law_pipeline.py all new       # 新法集
  python data/law_pipeline.py fetch|clean|chunk|embed [sfjs|new]
"""
import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import flk

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "laws"
OUT.mkdir(exist_ok=True)

# ── 语料集配置 ──────────────────────────────────────────────────
# 换一部法只改这里，下面的算法不动。
SETS = {
    "sfjs": {
        "名称": "司法解释",
        "laws": [
            {"简称": "合同编通则解释", "bbbs": "ff8081818c24e05b018c814e6de45ab5",
             "全称": "最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释"},
            {"简称": "总则编解释", "bbbs": "ff80818181cdceb30181d2c0af9e2339",
             "全称": "最高人民法院关于适用《中华人民共和国民法典》总则编若干问题的解释"},
            {"简称": "婚姻家庭编解释一", "bbbs": "ff80808177e75f880178006736f21aaf",
             "全称": "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）"},
            {"简称": "继承编解释一", "bbbs": "ff80808177e75f880178006a30861ac3",
             "全称": "最高人民法院关于适用《中华人民共和国民法典》继承编的解释（一）"},
        ],
    },
    "new": {
        "名称": "新法",
        "laws": [
            # 模型知识截止之后才颁布的 —— 无 RAG 必然答不出
            {"简称": "医疗保障法", "bbbs": "5d298a33dc474da595bf956cec680163",
             "全称": "中华人民共和国医疗保障法"},            # 2026-08-28，距今 17 天
            {"简称": "生态环境法典", "bbbs": "96630961659b4d87a65b7b1c595097fa",
             "全称": "中华人民共和国生态环境法典"},          # 2026-03-12
        ],
    },
}

# 正则
ART_HEAD = re.compile(r"^(第[一二三四五六七八九十百千零]+条)(.*)$")   # 行首条号 + 同行残余正文
ART_BRANCH = re.compile(r"^(第[一二三四五六七八九十百千零]+条)之[一二三四五六七八九十]+")  # 刑法式"条之一"
DIV_HEAD = re.compile(r"^第[一二三四五六七八九十百千零]+(?:编|分编|章|节)")  # 编/章/节 标题行
PAGE_NUM = re.compile(r"^[－\-—\s]*\d+[－\-—\s]*$")                   # 页脚：1 / －1－ / —1—

_CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn2int(s):
    s = str(s).lstrip("第").rstrip("条")
    total = section = num = 0
    for ch in s:
        if ch in _CN:
            num = _CN[ch]
        elif ch in _UNIT:
            section += (num or 1) * _UNIT[ch]
            num = 0
    return total + section + num


# ---- 阶段 1：抓取 ----
def stage_fetch(key):
    for law in SETS[key]["laws"]:
        dst = OUT / f"{law['简称']}_raw.txt"
        if dst.exists():
            print(f"[fetch] {law['简称']} 已存在，跳过")
            continue
        print(f"[fetch] {law['简称']} …")
        lines, pages = flk.fetch_text(law["bbbs"], verbose=False)
        dst.write_text("\n".join(lines), encoding="utf-8")
        print(f"        {pages} 页 / {len(lines)} 行 → {dst.name}")


# ---- 阶段 2：清洗 + 按条重排 ----
def stage_clean(key):
    for law in SETS[key]["laws"]:
        src = OUT / f"{law['简称']}_raw.txt"
        dst = OUT / f"{law['简称']}_articles.txt"
        lines = src.read_text(encoding="utf-8").splitlines()

        # 去页脚数字和空行
        kept = [l.strip() for l in lines if l.strip() and not PAGE_NUM.match(l.strip())]

        # 正文从"第一条"开始，前面是标题区和目录，砍掉。
        # 不能只找"第一个第X条"：正文里引用别的法（"依照民法典第五百八十七条"）也落在行首，
        # 得精确匹配"第一条"。
        start = next((i for i, l in enumerate(kept)
                      if (m := ART_HEAD.match(l)) and _cn2int(m.group(1)) == 1), None)
        if start is None:
            raise SystemExit(f"[clean] {law['简称']} 找不到'第一条'，检查抓取结果")
        body = kept[start:]

        # 拼断行 + 切条，编/章/节标题行是结构标题，跳过。
        # 切条按"条号必须连续"判：第 N 条后面只能是第 N+1 条。
        # 只按"行首是第X条"切会误切，因为正文里到处是"依照民法典第五百八十七条"这种引用，
        # 也顶在行首。实测司法解释这么切多出来 4 条。
        # 断行处直接 join 不加空格，中文本来就没空格，英文加了会把单词劈开。
        arts, cur_no, cur_body, dropped = [], None, [], 0
        for l in body:
            # "第X条之一" 必须在切条之前拦住：切条判据是"条号必须连续"，
            # 而"第十条之一"的条号也算成 10，不连续就没切，正文被静默并进"第十条"，
            # 下面那几道 gaps 校验照样通过 —— 内容丢了却一声不响。现有 7 部法没这种写法。
            if (mb := ART_BRANCH.match(l)):
                raise SystemExit(
                    f"[clean] {law['简称']} 出现「{mb.group(0)}」：它和「{mb.group(1)}」是两条不同的条文，"
                    f"当前按「条号必须连续」切条会把它的正文并进上一条。"
                    f"要加含「之一」的法，先让切条支持这种写法。")
            if DIV_HEAD.match(l) and not ART_HEAD.match(l):
                dropped += 1
                continue
            m = ART_HEAD.match(l)
            is_next = m and _cn2int(m.group(1)) == ((_cn2int(cur_no) + 1) if cur_no else 1)
            if is_next:
                if cur_no:
                    arts.append((cur_no, "".join(cur_body)))
                cur_no, cur_body = m.group(1), [m.group(2)]
            else:
                cur_body.append(l)
        if cur_no:
            arts.append((cur_no, "".join(cur_body)))

        dst.write_text("\n".join(f"{no}\n{b}" for no, b in arts), encoding="utf-8")

        # ── 校验 ──
        empty = [no for no, b in arts if not b.strip()]
        show  = [no for no, b in arts if len(b.strip()) < 5]
        nums  = [_cn2int(no) for no, _ in arts]
        gaps  = [nums[i] for i in range(1, len(nums)) if nums[i] != nums[i - 1] + 1]
        print(f"[clean] {law['简称']}: {len(arts)} 条 | 跳过标题行 {dropped} | 空正文 {len(empty)}"
              f" | 过短 {len(show)} | 条号 {nums[0]}–{nums[-1]}")
        if not arts:
            raise SystemExit(f"[clean] {law['简称']} 一条都没切出来")
        if empty:
            raise SystemExit(f"[clean] {law['简称']} 有空正文条：{empty[:5]} —— 切分规则出错")
        if gaps:
            raise SystemExit(f"[clean] {law['简称']} 条号不连续，断点 {gaps[:8]} —— 切分规则出错")
        if show:
            print(f"         ⚠ 过短正文（人工抽查）：{show[:5]}")


# ---- 阶段 3：分块 ----
def stage_chunk(key):
    cfg = SETS[key]
    all_chunks = []
    for law in cfg["laws"]:
        src = OUT / f"{law['简称']}_articles.txt"
        lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 文件是"条号一行、正文一行"交替
        for i in range(0, len(lines) - 1, 2):
            no, body = lines[i], lines[i + 1]
            if not no.startswith("第"):
                continue
            # 检索文本 = 法律名 + 条号 + 正文，喂给 embedding 的就是它
            all_chunks.append({
                "law": law["全称"], "简称": law["简称"],
                "编": "", "分编": "", "章": "", "节": "",
                "条号": no, "正文": body,
                "检索文本": f"《{law['全称']}》{no} {body}",
            })

    dst = OUT / f"{cfg['名称']}_chunks.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    per = {}
    for c in all_chunks:
        per[c["简称"]] = per.get(c["简称"], 0) + 1
    print(f"[chunk] {cfg['名称']} 合计 {len(all_chunks)} chunk → {dst.name}")
    for k, v in per.items():
        print(f"        {k}: {v} 条")
    if len(all_chunks) < 30:
        raise SystemExit("[chunk] chunk 数太少，前面的抓取/清洗有问题")


# ---- 阶段 4：编码成向量 ----
def stage_embed(key):
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 必须在 import 模型库之前
    import numpy as np
    from sentence_transformers import SentenceTransformer

    name = SETS[key]["名称"]
    src = OUT / f"{name}_chunks.jsonl"
    dst = OUT / f"{name}_vectors.npy"
    chunks = [json.loads(l) for l in open(src, encoding="utf-8")]
    print(f"[embed] {name} 待编码 {len(chunks)} 条")

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vecs = np.asarray(model.encode([c["检索文本"] for c in chunks],
                                   batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False), dtype="float32")
    np.save(dst, vecs)

    # 向量行数必须等于 chunk 数。对不上就是错位，而且不报错，
    # 只会在后面检索时把向量接到错的条文上。
    if vecs.shape[0] != len(chunks):
        raise SystemExit(f"[embed] 行序契约被打破：{vecs.shape[0]} 行 vs {len(chunks)} 条")
    print(f"[embed] {vecs.shape} → {dst.name}")


STAGES = {"fetch": stage_fetch, "clean": stage_clean, "chunk": stage_chunk, "embed": stage_embed}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    stage = args[0] if args and args[0] in list(STAGES) + ["all"] else "all"
    key   = args[1] if len(args) > 1 else (args[0] if args and args[0] in SETS else "sfjs")
    todo  = list(STAGES) if stage == "all" else [stage]
    print(f"### 语料集 {key}（{SETS[key]['名称']}）| 阶段 {todo}")
    for s in todo:
        STAGES[s](key)
