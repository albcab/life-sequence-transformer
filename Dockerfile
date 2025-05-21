###albcab/whip:gpu
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04


FROM albcab/whip:gpus

# ENV CUDA_HOME=/usr/local/cuda-12.4
# RUN add-apt-repository ppa:deadsnakes/ppa && \
#     apt install -y python3.12-dev

#whip2vec files
COPY figaro/src/ ./w2v/src/
COPY figaro/conf/ ./w2v/conf/
COPY figaro/requirements.txt .
COPY figaro/run.sh .
COPY figaro/run_experiments.sh ./w2v/
COPY figaro/*_test_ids* ./w2v/
COPY figaro/*_idx_cache.json ./w2v/
COPY figaro/counter.py ./w2v/
COPY figaro/income_100_cache.json ./w2v/

#life2vec files
COPY life2vec-light/src/ ./l2v/src/
COPY life2vec-light/conf/ ./l2v/conf/

# RUN python3.12 -m pip install --no-cache-dir -r requirements.txt
# RUN python3.12 -m pip install --no-cache-dir pytorch-fast-transformers==0.4.0

CMD ["/bin/bash"]




# # FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# # FROM ubuntu:22.04
# LABEL maintainer="whip2whip"
# LABEL repository="figaro"

# ENV DEBIAN_FRONTEND=noninteractive

# ENV LC_ALL=it_IT.ISO-8859-1
# ENV LANG=it_IT.ISO-8859-1
# ENV LANGUAGE=it_IT.ISO-8859-1

# ENV CUDA_HOME=/usr/local/cuda-12.4

# RUN apt update && apt install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt update && \
#     apt install -y bash \
#                    build-essential \
#                    git \
#                    git-lfs \
#                    curl \
#                    ca-certificates \
#                    vim \
#                    python3.12 \
#                    python3-pip \
#                    python3.12-dev \
#                    python3.12-venv && \
#     rm -rf /var/lib/apt/lists

# WORKDIR /usr/src/

# #must run from Project-WHIP
# #whip2vec files
# COPY figaro/src/ ./w2v/src/
# COPY figaro/conf/ ./w2v/conf/
# COPY figaro/requirements.txt .
# COPY figaro/run.sh .

# #life2vec files
# COPY life2vec-light/src/ ./l2v/src/
# COPY life2vec-light/conf/ ./l2v/conf/

# # make sure to use venv
# RUN python3.12 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
#     python3.12 -m pip install --no-cache-dir \
#         torch==2.5.1 \
#         torchvision==0.20.1 \
#         torchaudio==2.5.1 \
#         # --extra-index-url https://download.pytorch.org/whl/cu118 && \
#         # --extra-index-url https://download.pytorch.org/whl/cpu && \
#         --extra-index-url https://download.pytorch.org/whl/cu124 && \
#     python3.12 -m pip install --no-cache-dir -r requirements.txt
# RUN python3.12 -m pip install --no-cache-dir pytorch-fast-transformers==0.4.0

# CMD ["/bin/bash"]
