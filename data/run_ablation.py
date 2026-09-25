# -*- coding: utf-8 -*-
"""
一次性运维脚本：把 n=100 的完整消融一口气跑完，关掉 VSCode 也能继续跑。

bge-reranker-v2-m3 在 CPU 上约 20 秒/题，两个 rerank 模式 × 三个语料 = 600 题，
三个多小时，普通会话撑不到那么久。用 PowerShell 的 Start-Process 单独起一个进程，
父进程立刻退出，窗口藏起来，VSCode 关掉它还在跑。

用法（项目根目录执行）：
  powershell -NoProfile -Command "Start-Process 'D:\\python3.15\\python.exe' `
    -ArgumentList '-X','utf8','data/run_ablation.py' `
    -WorkingDirectory 'C:\\Users\\hujinghao\\Desktop\\Agent study\\legal_RAG' -WindowStyle Hidden"

看进度：tail -f data/run_log.txt（每行立刻落盘，随时能 tail）

结果落在 data/评测结果.jsonl，检索和端到端都追加到那儿。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 必须在 import 模型库之前
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")                # Xet 协议走不通镜像，401

ROOT = Path(__file__).parent.parent      # 本脚本在 data/，上一级是项目根
sys.path.insert(0, str(ROOT / "src"))
LOG_PATH = ROOT / "data" / "run_log.txt"


class Tee:
    """同时写控制台和日志文件，每行立刻 flush，随时能 tail 看进度。
    隐藏窗口启动时 sys.__stdout__ 可能是 None，所以过滤掉空流。"""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for f in self.streams:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.streams:
            f.flush()


def main():
    log = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log)

    print(f"\n===== 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)

    import eval_recall as er
    import eval_answer as ea

    # ── 第一段：rerank 扫描，最慢，先跑 ──
    print("### 第一段：rerank 扫描（v2-m3）", flush=True)
    for c in ["minfadian", "laws", "new"]:
        t = time.time()
        er.main(er.RERANK_SWEEP, c, True, None)
        print(f"  [{c} 用时 {time.time() - t:.0f}s]", flush=True)

    # ── 第二段：端到端 no-rag 基线，纯 API 不占 CPU ──
    print("### 第二段：端到端 no-rag", flush=True)
    for c in ["minfadian", "laws", "new"]:
        t = time.time()
        ea.main("no-rag", c, use_rewrite=False, retr_mode="dense")
        print(f"  [{c} 用时 {time.time() - t:.0f}s]", flush=True)

    print(f"===== 完成 {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)


if __name__ == "__main__":
    main()
