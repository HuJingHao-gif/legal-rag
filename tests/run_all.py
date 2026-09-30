# -*- coding: utf-8 -*-
"""
一条命令跑完 tests/ 下的所有测试。

为什么需要它：测试文件一多，就得手动一个个跑。漏跑一个和"全绿"看起来是一样的，
而漏掉的往往正是会红的那个 —— 上一批就是这么漏掉 test_textutil 的。

用 glob 自动发现、不写死列表：新增测试不用改这里；改了忘了同步也不会静默漏跑。
按文件名排序是为了让输出顺序稳定，方便跟上次的结果对着看。

运行: python tests/run_all.py
"""
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
ROOT = TESTS_DIR.parent


def main():
    files = sorted(TESTS_DIR.glob("test_*.py"))
    if not files:
        print(f"在 {TESTS_DIR} 下没找到 test_*.py —— 路径不对，或测试被挪走了")
        return 1

    # 子进程强制 UTF-8：测试里打了 ✓ 和中文，一旦把输出重定向到文件
    # （data/ 下那些 run_stdout.txt 就是这么来的），stdout 就不是控制台了，
    # 编码会退回系统默认的 cp936，✓ 编不出来直接 UnicodeEncodeError。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    failed = []
    for f in files:
        print(f"\n{'=' * 70}\n{f.name}\n{'=' * 70}")
        # cwd 固定成项目根：测试里读的是 data/ 这类相对路径，
        # 从别的目录调用 run_all.py 时不能跟着调用者的当前目录走。
        proc = subprocess.run([sys.executable, str(f)], cwd=ROOT, env=env)
        if proc.returncode != 0:
            failed.append((f.name, proc.returncode))

    print(f"\n{'=' * 70}")
    if failed:
        print(f"FAIL ({len(failed)}/{len(files)} files)")
        for name, rc in failed:
            print(f"  - {name}（退出码 {rc}）")
        return 1
    print(f"ALL PASS ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
