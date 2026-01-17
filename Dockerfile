FROM continuumio/miniconda3

WORKDIR /workspace

# 先拷贝环境定义
COPY environment.yml /workspace/

# 用 conda 创建同名环境并清理缓存
RUN conda env create -f environment.yml && \
    conda clean -afy

# 默认进入容器时激活 hybrid_sd
RUN echo "conda activate hybrid_sd" >> ~/.bashrc

# 拷贝项目代码
COPY . /workspace

# 调试友好：默认进入 bash，如需直接运行脚本可改成 python main.py 等
CMD ["/bin/bash"]

