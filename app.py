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
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from answer import ask, ask_stream, warmup, CORPORA          # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时先把默认语料的索引和模型热好。不热的话第一个问题要等 40~60 秒，
    # 页面看着就像卡死了。这里同步做完再开始收请求，理由见 answer.warmup 的注释。
    warmup()
    yield


app = FastAPI(title="法律 RAG 问答", description="带引用的法律问答，可核验", lifespan=lifespan)


class AskReq(BaseModel):
    q: str
    corpus: str = "all"
    k: int = 5
    mode: str = "hybrid"          # 默认不挂 rerank，理由见 src/retrieve.py 里 RERANK_NAME 那段
    rewrite: bool = True


# 必须和 src/retrieve.py 的 search() 对齐。放在这里做入口校验，是因为
# search() 认不出 mode 时抛的是 SystemExit —— 它继承 BaseException 不继承 Exception，
# 在服务器里能一路穿出去把 uvicorn 进程带走（10-10 真踩到，见下面 gen() 的注释）。
MODES = {"dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank"}


def _bad_request(req):
    if req.corpus not in CORPORA:
        return f"未知语料 {req.corpus}，可选：{list(CORPORA)}"
    if req.mode not in MODES:
        return f"未知检索模式 {req.mode}，可选：{sorted(MODES)}"
    return None


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
    bad = _bad_request(req)
    if bad:
        return {"error": bad}
    return ask(req.q, corpus=req.corpus, k=req.k, mode=req.mode, use_rewrite=req.rewrite)


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream_api(req: AskReq):
    """流式问答，SSE。页面走这个；/ask 留着给脚本和 curl 用，不替换它。

    用 POST + fetch 读流，不用 EventSource —— EventSource 只发 GET，
    塞不下问题加那几个参数。
    """
    bad = _bad_request(req)
    if bad:
        # 参数错同步就能判出来，直接 400，别包成 SSE 事件让页面猜
        raise HTTPException(400, bad)

    def gen():
        try:
            for event, data in ask_stream(req.q, corpus=req.corpus, k=req.k,
                                          mode=req.mode, use_rewrite=req.rewrite):
                yield _sse(event, data)
        except BaseException as e:
            # 这里必须是 BaseException，不能是 Exception。
            # 10-10 踩到：mode 传错时 search() 抛 SystemExit，而 SystemExit 继承
            # BaseException，except Exception 接不住 —— 异常穿过流式响应把整个
            # uvicorn 进程干掉了，一次错请求 = 服务全挂。服务器里单个请求的错
            # 绝不能升级成进程级的错。
            # 注意这不是"吞掉"：错误照样以 error 事件发给页面，照样看得见。
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def index():
    """单文件页面直接内联返回，省得配静态目录"""
    return (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    # 默认 127.0.0.1：本地开发不把服务暴露到局域网。
    # 容器里必须用 HOST=0.0.0.0，否则端口映射（-p 8000:8000）进不来。
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
