# -*- coding: utf-8 -*-
"""
FastAPI 服务：把 src/answer.py 包成 HTTP 接口。

分三层，底下那层才是资产：
    核心逻辑  src/answer.py
    接口      app.py
    壳        index.html

跑起来：
    python app.py
    → http://127.0.0.1:8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from answer import ask, CORPORA          # noqa: E402

app = FastAPI(title="法律 RAG 问答", description="带引用的法律问答，可核验")


class AskReq(BaseModel):
    q: str
    corpus: str = "all"
    k: int = 5
    mode: str = "hybrid"          # 默认不挂 rerank，理由见 src/retrieve.py 里 RERANK_NAME 那段
    rewrite: bool = True


@app.get("/corpora")
def corpora():
    """前端用它渲染语料下拉框"""
    return {k: v["名称"] for k, v in CORPORA.items()}


@app.get("/presets")
def presets(corpus: str = "all", n: int = 6):
    """预设问题，直接从评测集里抽。

    这样页面上看到的演示效果，和消融表里的数字说的是同一批题。
    评测集 09-22 换成 _v2 了（100 题/语料），这里得跟着改，漏了就两边对不上。
    all 用新法的题 —— 模型不知道这批法，演示时最能看出检索在起作用。
    """
    import json
    eval_file = {"minfadian": "评测集_v2.jsonl", "laws": "司法解释_评测集_v2.jsonl",
                 "new": "新法_评测集_v2.jsonl", "all": "新法_评测集_v2.jsonl"}.get(corpus)
    path = ROOT / "data" / eval_file if eval_file else None
    if not path or not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 口语化题排前面，演示效果好；不够再用原话题凑
    ko = [r["问题"] for r in rows if r["类型"] == "口语化"]
    yq = [r["问题"] for r in rows if r["类型"] == "法条原话"]
    return (ko + yq)[:n]


@app.post("/ask")
def ask_api(req: AskReq):
    if req.corpus not in CORPORA:
        return {"error": f"未知语料 {req.corpus}，可选：{list(CORPORA)}"}
    return ask(req.q, corpus=req.corpus, k=req.k, mode=req.mode, use_rewrite=req.rewrite)


@app.get("/", response_class=HTMLResponse)
def index():
    """单文件页面直接内联返回，省得配静态目录"""
    return (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
