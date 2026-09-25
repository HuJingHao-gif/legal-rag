# -*- coding: utf-8 -*-
"""
从条文反向生成评测题候选，再交人工筛，最后定稿 30~50 题。
每层抽样一批条文，让 DeepSeek 站在"普通人 / 律师"两个视角各问一句。

输入输出都走命令行（默认是民法典）。
用法：
  setx DEEPSEEK_API_KEY "sk-xxxx"，然后重启 VSCode —— 环境变量要重启才生效
  python make_eval.py                                  # 民法典
  python make_eval.py <chunks.jsonl> <out.jsonl> <每层抽样数>
"""
import os
import re
import sys
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import anthropic

DATA_DIR = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "民法典_chunks.jsonl"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "评测集_候选.jsonl"

PER_BANK = int(sys.argv[3]) if len(sys.argv) > 3 else 10   # 每个"分层"抽多少条
BATCH    = 2     # 每次 API 调用塞几条
MAX_RETRY = 2    # 一批失败重试几次

# 坑（实测）：deepseek-v4-flash 是推理模型，先吐一大段 thinking 才给正文。
# max_tokens 设 2048 时 token 全被推理吃掉，正文被截断，JSON 就残了。
# 现在给 8192，每批只塞 2 条。
MAX_TOKENS = 8192

# ── 连接：密钥不进代码 ────────────────────────────────────────
# 先读环境变量 DEEPSEEK_API_KEY，没有再看 data/api_key.txt（一行只放 key）。
# 写进 .py 就等于作废，传到 Gitee 会被爬虫秒扫。.py 要提交，api_key.txt 进 .gitignore。
KEY_FILE = DATA_DIR / "api_key.txt"
api_key = os.environ.get("DEEPSEEK_API_KEY") or (
    KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else None
)
if not api_key:
    raise SystemExit(
        "没找到 API key。二选一：\n"
        '  ① setx DEEPSEEK_API_KEY "sk-..."  然后重启 VSCode\n'
        "  ② 把 key 单独一行存到 data/api_key.txt（并确保它进 .gitignore）"
    )

client = anthropic.Anthropic(
    base_url="https://api.deepseek.com/anthropic",
    api_key=api_key,
)

# 出题提示词。问题里不许出现条文原词、不许提条号，否则测不出"词汇不匹配"
PROMPT = """你是法律问答数据集的设计者。下面给你若干条中国法条文，请为**每一条**设计 2 个问题：

1. "口语化"：一个完全不懂法律的普通人，遇到这件事时会怎么问。
   —— 必须用日常口语，**不许照搬条文里的生僻词**，更不许出现"第X条"。
2. "法条原话"：律师引用法条原文的问法（保留关键法律术语，但仍是问句）。

要求：
- 每个问题要**只靠这一条就能回答**，别设计需要跨条才能答的。
- 若某条**不适合出题**（如"本法自X日起施行""本法所称X，包括…"这类纯定义/程序条款），
  把 suitability 标成 false，questions 留空数组。
- 只输出 JSON 数组，不要任何解释。格式：
[{"idx": 0, "suitability": true,
  "questions": [{"类型": "口语化", "问题": "..."}, {"类型": "法条原话", "问题": "..."}]}]

待出题的条文：
%s
"""


def load_chunks():
    return [json.loads(l) for l in open(src, encoding="utf-8")]


def bank_of(c):
    """分层键：民法典按"编"分，司法解释没有编，退回按"简称"（法律名）分。"""
    return c.get("编") or c.get("简称") or "全部"


def loc_of(c):
    """给模型看的位置说明（民法典给编/章路径，司法解释给法律名）"""
    path = " ".join(x for x in (c.get("编"), c.get("分编"), c.get("章"), c.get("节")) if x)
    return path or c.get("简称") or c.get("law", "")


def stratified_sample(chunks):
    """分层抽样：每层各抽几条，避免题目全挤在一层（那样评测集就偏了）"""
    by_bank = defaultdict(list)
    for c in chunks:
        by_bank[bank_of(c)].append(c)
    picked = []
    for bank, items in by_bank.items():
        picked += random.sample(items, min(PER_BANK, len(items)))
    return picked


def extract_json(text):
    """模型爱在 JSON 外面套 ``` 或者加句废话，只把 [...] 那段抠出来"""
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        raise ValueError(f"没找到 JSON 数组，原文前 200 字：{text[:200]}")
    return json.loads(m.group(0))


def ask_llm(batch):
    """把一批条文丢给 DeepSeek，返回它给的 JSON 数组"""
    listing = "\n\n".join(
        f'[idx {i}] 条号：{c["条号"]}｜出处：{loc_of(c)}\n正文：{c["正文"]}'
        for i, c in enumerate(batch)
    )
    msg = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT % listing}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")   # 只取 text 块，thinking 丢
    return extract_json(text)


def ask_with_retry(batch):
    """一批失败就重试；推理模型偶尔会吐出不合法 JSON，重试通常就好了"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return ask_llm(batch)
        except Exception as e:
            if attempt == MAX_RETRY:
                raise
            print(f"    重试 {attempt}/{MAX_RETRY - 1}…（{type(e).__name__}: {str(e)[:60]}）")
            time.sleep(2)


def main():
    random.seed(42)                          # 固定种子，重跑拿到同一批题
    chunks = load_chunks()
    picked = stratified_sample(chunks)
    print(f"分层抽样：{len(picked)} 条待出题")

    rows, qid = [], 0
    for start in range(0, len(picked), BATCH):
        batch = picked[start:start + BATCH]
        try:
            result = ask_with_retry(batch)
        except Exception as e:
            print(f"  第 {start//BATCH+1} 批失败：{e}")
            continue
        for item in result:
            if not item.get("suitability"):
                continue
            idx = item.get("idx", -1)
            if not (0 <= idx < len(batch)):  # 防模型编出越界的 idx
                continue
            c = batch[idx]
            for q in item.get("questions", []):
                qid += 1
                rows.append({
                    "id": f"q{qid:03d}",
                    "问题": q["问题"],
                    "类型": q["类型"],
                    "标准条号": [c["条号"]],
                    "标准法律": c.get("law", ""),
                    "备注": loc_of(c),
                })
        print(f"  第 {start//BATCH+1} 批：累计候选 {len(rows)} 条")
        time.sleep(1)                        # 别把 API 打太狠

    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")   # 中文原样存，方便肉眼看

    print(f"\n候选问题已存：{dst}（{len(rows)} 条）")
    print("★ 下一步（人工）：打开文件，去重 / 去歧义 / 去太偏，筛到 30~50 条，另存为 评测集.jsonl")


if __name__ == "__main__":
    main()
