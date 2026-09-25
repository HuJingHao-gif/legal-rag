# Day1 练习：手写 agent 循环。
# 想跑通的是那件事——模型只负责"说要调什么工具"，真正执行的是我的代码。
import os
from pathlib import Path

import anthropic  # 官方 SDK

# 密钥不进代码：环境变量优先，没有就退到 data/api_key.txt
# （09-22 之前这里明文写着一个 key，已作废重发）
_key = os.environ.get("DEEPSEEK_API_KEY") or (
    Path(__file__).parent.parent / "data" / "api_key.txt"
).read_text(encoding="utf-8").strip()

client = anthropic.Anthropic(
    base_url="https://api.deepseek.com/anthropic",  # DeepSeek 的 Anthropic 兼容端点
    api_key=_key,
)

# 工具说明书（JSON，给模型看的"操作手册"）
tools = [
    {
        "name":"get_weather",
        "description":"查询某个城市的当前天气",
        "input_schema":{
            "type":"object",
            "properties":{
                "city": {"type":"string","description":"城市名，例如 北京"}
            },
            "required":["city"],
        },
    }
]

# 真实函数：工具被调用时跑的是这段
def get_weather(city:str)->str:
    return f"{city}今天晴,25℃,适合学习"

# 工具名 → 真实函数的查找表
TOOLS = {"get_weather": get_weather}

# agent 循环
messages = [
    {"role": "user", "content": "北京今天天气怎么样？"}
]

while True:
    response = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        break  # 模型说完了，跳出

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"  → 模型要调工具：{block.name}({block.input})")
            result = TOOLS[block.name](**block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})

# 输出答案
for block in response.content:
    if block.type == "text":
        print("最终答案：", block.text)
