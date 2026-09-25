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


def ask_json(prompt, retry=2):
    """要模型输出 JSON。它偶尔会套 ``` 或加句废话，先正则抠出 [...] / {...} 再 parse。"""
    for attempt in range(1, retry + 1):
        try:
            text = ask(prompt, retry=1)
            m = re.search(r"[\[{].*[\]}]", text, flags=re.S)
            if not m:
                raise ValueError(f"没找到 JSON。原文前 120 字：{text[:120]}")
            return json.loads(m.group(0))
        except Exception:
            if attempt == retry:
                raise
            time.sleep(2)
