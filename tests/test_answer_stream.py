# -*- coding: utf-8 -*-
"""
流式问答的两个判据测试：从半截 JSON 里抠 answer，以及事件序列的契约。

为什么单独测这个：流式路上模型输出的还是 JSON，页面要的是里面的 answer 字段，
中间隔着一段"边收边切"的代码（_partial_answer）。它**每次只拿到半截输入**，
而切点在哪儿完全随机 —— 可能正好落在 \\" 或 \\u 中间。这段代码写错了不会崩，
只会让页面上少几个字、或多几个乱码字，正是这个项目最怕的静默失败。

所以拿随机分片喂它，验两条不变式：
  1) 过程中吐出来的每一段都必须是最终答案的前缀（不能凭空造字、不能跳字）
  2) 收完之后必须恰好等于完整答案

另外验事件序列：meta → delta* → done；以及只收到半截就断了必须抛 ——
半截正文不能当成品交出去，那和"伪造一个看着正常的默认值"是一回事。

运行: python tests/test_answer_stream.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import answer
from answer import _partial_answer

SEED = 20261010          # 定种子：随机分片要能复现，不然红了都不知道怎么重跑


def hit(no="第二十三条", law="医疗保障法"):
    """造一条假的"检索到的资料"。返回 (分数, chunk) —— 跟 search() 的输出同形。"""
    return (0.9, {"条号": no, "简称": law, "正文": "（条文正文，测试里不重要）"})


# ---------------------------------------------------------------- 切 JSON

CASES = [
    ("普通答案",              {"answer": "你好", "citations": []}),
    ("答案里有转义符",         {"answer": '带"引号"和反斜杠\\以及换行\n第二行', "citations": []}),
    ("空答案",                {"answer": "", "citations": []}),
    ("中文标点",              {"answer": "《民法典》第1241条；它并未规定……", "citations": ["《民法典》第1254条"]}),
    ("citations 排在前面",    {"citations": ["《民法典》第1254条"], "answer": "字段顺序不一定"}),
]


def check_partial_answer():
    rnd = random.Random(SEED)
    ok = bad = 0
    for desc, obj in CASES:
        full = json.dumps(obj, ensure_ascii=False)
        want = obj["answer"]
        problems = []
        shown = None
        for _ in range(200):
            buf = ""
            for piece in split_random(full, rnd):
                buf += piece
                got = _partial_answer(buf)
                if got is None:
                    continue
                if not want.startswith(got):
                    problems.append(f"中途吐出的不是最终答案的前缀：{got!r}")
                    break
                shown = got
        if not problems and shown != want:
            problems.append(f"收完了却对不上，期望 {want!r}，实际 {shown!r}")

        # 关键字还没出现时必须返回 None，不能返回空串冒充"答案是空的"
        if _partial_answer('{"cita') is not None:
            problems.append("answer 字段还没出现就返回了值")

        if problems:
            bad += 1
            print(f"  ✗ {desc}")
            for p in problems:
                print(f"      {p}")
        else:
            ok += 1
            print(f"  ✓ {desc}（200 次随机分片）")
    return ok, bad


def split_random(text, rnd):
    """把整段 JSON 随机切成 1~4 字一块 —— 逼着切点落在转义符和引号中间"""
    out, i = [], 0
    while i < len(text):
        n = rnd.randint(1, 4)
        out.append(text[i:i + n])
        i += n
    return out


# ---------------------------------------------------------------- 事件序列

def fake_llm(pieces, boom=None):
    def gen(prompt, retry=2):
        for p in pieces:
            yield p
        if boom:
            raise boom
    return gen


def stream(pieces):
    """把重家伙（模型、检索）换掉，只走流式那段逻辑"""
    answer._index = lambda c: ([], None, None)
    answer.rw.expand = lambda q: q
    answer.search = lambda q, **kw: [hit()]
    answer.llm.ask_stream = fake_llm(pieces)
    return list(answer.ask_stream("问题", use_rewrite=False))


def check_events():
    ok = bad = 0

    def report(desc, problems):
        nonlocal ok, bad
        if problems:
            bad += 1
            print(f"  ✗ {desc}")
            for p in problems:
                print(f"      {p}")
        else:
            ok += 1
            print(f"  ✓ {desc}")

    # ① 正常一轮：meta → delta* → done，且每个 delta 都是最终答案的前缀
    evs = stream(['{"ans', 'wer": "张三", "cita', 'tions": ["《医疗保障法》第23条"]}'])
    names = [e for e, _ in evs]
    done = evs[-1][1]
    problems = []
    if names[-1] != "done":
        problems.append(f"最后一个事件不是 done：{names[-1]}")
    if names[0] != "meta" or names[1] != "delta":
        problems.append(f"开头不是 meta→delta：{names[:3]}")
    if done["answer"] != "张三":
        problems.append(f"done 里的答案不对：{done['answer']!r}")
    if not done["引用校验"]["通过"]:
        problems.append(f"引用校验不该失败：{done['引用校验']}")
    deltas = [d["text"] for e, d in evs if e == "delta"]
    if any(not done["answer"].startswith(t) for t in deltas):
        problems.append(f"有 delta 不是最终答案的前缀：{deltas}")
    report("正常一轮：meta → delta* → done，delta 都是答案前缀", problems)

    # ② 只收到半截就断了：必须抛。安静地把半截当成品，就是伪造一个看着正常的结果
    try:
        stream(['{"answer": "这是半截答案'])
        report("半截就断了必须抛", ["没抛 —— 残缺正文被当成成品交出去了"])
    except Exception as e:
        report("半截就断了必须抛", [] if "JSON" in str(e) else [f"抛的错不对：{e!r}"])

    # ③ 吐到一半 API 挂了：同样必须抛，而不是把已推出的那部分当答案
    try:
        answer.llm.ask_stream = fake_llm(['{"answer": "前半句'], boom=RuntimeError("API 500"))
        list(answer.ask_stream("问题", use_rewrite=False))
        report("中途抛异常必须往上传", ["没抛 —— 错误被吞了"])
    except RuntimeError:
        report("中途抛异常必须往上传", [])
    except Exception as e:
        report("中途抛异常必须往上传", [f"抛的不是原始异常：{e!r}"])

    return ok, bad


def main():
    print("一、从半截 JSON 里抠 answer")
    ok1, bad1 = check_partial_answer()
    print(f"\n二、事件序列契约（{ok1} 项已过）")
    ok2, bad2 = check_events()

    ok, bad = ok1 + ok2, bad1 + bad2
    print(f"\n{ok}/{ok + bad} 通过")
    if bad:
        print(f"★ 有 {bad} 个用例失败 —— 流式路径的判据有问题")
        return 1
    print("流式路径的切分与事件契约都对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
