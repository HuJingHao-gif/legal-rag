# 法律 RAG 问答服务
FROM python:3.13-slim

# deb.debian.org 在国内会把构建卡死（实测 9.6MB 的包索引 9 分钟没下完），换清华源
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HOST/PORT 见 app.py 末尾；HF_HOME 是运行时挂进来的模型缓存
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/hf-cache \
    HOST=0.0.0.0 \
    PORT=8000

# 先装依赖再拷代码：requirements.txt 不变时改代码不会触发重装
COPY requirements.txt .

# torch 必须单独走 CPU 源：PyPI 的 Linux 版会拖 CUDA 全家桶（6~10GB）。
# 用交大镜像 —— 它是标准 PEP503 索引；阿里那个同名仓是平铺目录，--index-url 会 404。
# grep 滤掉 requirements.txt 里那行 —— 不滤的话 pip 会认为 2.6.0 不满足
# >=2.13.0，转头去 PyPI 又把 CUDA 版拉回来。
# transformers 钉 5.17.0：5.19.0 的 get_device_type() 直接调
# torch.accelerator.current_accelerator()，而 torch 2.6 在没有加速器时是【抛异常】
# 而不是返回 None，于是启动即崩。5.17.0 没这个问题，也正好是本地开发环境的版本。
RUN pip install --no-cache-dir \
        --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/ \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        torch==2.6.0+cpu \
    && grep -v '^torch' requirements.txt > /tmp/req-notorch.txt \
    && pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /tmp/req-notorch.txt \
        "transformers==5.17.0"

COPY app.py index.html ./
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["python", "app.py"]
