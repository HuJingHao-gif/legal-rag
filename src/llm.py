# -*- coding: utf-8 -*-
"""
LLM 调用的公共层：密钥读取 + 带重试的调用 + JSON 解析。
eval_answer.py / rewrite.py 都走这里，省得每个脚本各写一遍密钥逻辑。

密钥不进 .py：环境变量 DEEPSEEK_API_KEY 优先，没有就退到 data/api_key.txt。
09-22 发现 practice/agent_loop.py 里还留着一个明文 key，作废重发后补掉了。
"""
import os
import re
import json
import time
from pathlib import Path

import anthropic

KEY_FILE = Path(__file__).parent.parent / "data" / "api_key.txt"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 8192       # v4-flash 是推理模型，先吐一大段 thinking 才给正文。
                        # 给小了 token 全被推理吃掉，正文截断，拿到残缺 JSON（踩过）
_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY") or (
            KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else None)
        if not key:
            raise SystemExit("没找到 API key（环境变量 DEEPSEEK_API_KEY 或 data/api_key.txt）")
        _client = anthropic.Anthropic(base_url="https://api.deepseek.com/anthropic", api_key=key)
    return _client


def ask(prompt, retry=2):
    """返回模型的正文文本，丢掉 thinking 块"""
    for attempt in range(1, retry + 1):
        try:
            msg = client().messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                           messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception:
            if attempt == retry:
                raise
            time.sleep(2)


def parse_json(text):
    """从模型输出里抠出 JSON。它偶尔会套 ``` 或加句废话。

    单独抽出来是因为流式那条路径也要用：边收边推的时候只切 answer 字段，
    等收完了走的还是这个函数 —— 两条路径解 JSON 的规则必须是同一个。
    """
    m = re.search(r"[\[{].*[\]}]", text, flags=re.S)
    if not m:
        raise ValueError(f"没找到 JSON。原文前 120 字：{text[:120]}")
    return json.loads(m.group(0))


def ask_json(prompt, retry=2):
    """要模型输出 JSON，走 parse_json。"""
    for attempt in range(1, retry + 1):
        try:
            return parse_json(ask(prompt, retry=1))
        except Exception:
            if attempt == retry:
                raise
            time.sleep(2)


def ask_stream(prompt, retry=2):
    """流式版 ask()：边生成边 yield 正文，thinking 块一样丢掉。

    messages.stream() 的 text_stream 只给 text_delta，reasoning 模型的
    thinking 段是 thinking_delta，进不来 —— 跟 ask() 里那个
    `if b.type == "text"` 是同一个口径。

    已经吐出去东西之后就不再重试：前半截答案已经发到浏览器上了，
    重来一遍会拼出两段接不上的话。这时候宁可抛出去，让页面显示"生成中断"。
    """
    for attempt in range(1, retry + 1):
        yielded = False
        try:
            with client().messages.stream(model=MODEL, max_tokens=MAX_TOKENS,
                                          messages=[{"role": "user", "content": prompt}]) as s:
                for text in s.text_stream:
                    yielded = True
                    yield text
            return
        except Exception:
            if yielded or attempt == retry:
                raise
            time.sleep(2)
