import requests
"""
def walk(node, depth=0):
    print("  " * depth + f"{node['title']}（{len(node['children'])}个子节点）")
    if node['children']:
        walk(node['children'][0], depth + 1)

def max_depth(node):
    if not node["children"]:      # 叶子：深度就是 0
        return 0
    deepest = 0
    for child in node["children"]:       # 遍历所有孩子
        d = max_depth(child)             # 问每个孩子："你最深到几层？"
        if d > deepest:                  # 记住最深的那个
            deepest = d
    return 1 + deepest                   # 我的深度 = 孩子最深 + 我自己这一层

def show(node, depth=0, limit=3):
    if depth >= limit:              # 只打到第 3 层，不刷屏
        return
    print("  " * depth + f"{node['title']}（{len(node['children'])}个孩子）")
    for child in node["children"]:
        show(child, depth + 1, limit)
"""
def find_first_tiao(node):
    if "条" in node["title"] and node["children"] == []:   # 叶子且标题含"条"
        return node
    for child in node["children"]:
        result = find_first_tiao(child)
        if result:
            return result
    return None

# 请求头和搜索接口一样，没改
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Referer": "https://flk.npc.gov.cn/detail?id=ff808081729d1efe01729d50b5c500bf",
    "Origin": "https://flk.npc.gov.cn",
}
def find_first_tiao(node):
    if "条" in node["title"] and node["children"] == []:   # 叶子且标题含"条"
        return node
    for child in node["children"]:
        result = find_first_tiao(child)
        if result:
            return result
    return None
# 详情接口是 GET，URL 和参数都跟搜索不一样
url = "https://flk.npc.gov.cn/law-search/search/flfgDetails"
params = {"bbbs": "ff808081729d1efe01729d50b5c500bf"}   # 上一步拿到的民法典 ID

resp = requests.get(url, headers=headers, params=params)   # GET + params=
print("HTTP 状态码：", resp.status_code)
data = resp.json()
print("业务状态码：", data.get("code"))
#print(data)

"""
walk(data["data"]["content"])
print("整棵树最大深度：", max_depth(data["data"]["content"]))
show(data["data"]["content"], limit=3)
"""
first = find_first_tiao(data["data"]["content"])
print(first)