###albcab/whip:gpu
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04
# FROM albcab/whip:gpu
# FROM ubuntu:22.04
LABEL maintainer="whip2whip"
LABEL repository="figaro"

ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=it_IT.ISO-8859-1
ENV LANG=it_IT.ISO-8859-1
ENV LANGUAGE=it_IT.ISO-8859-1

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   vim \
                   python3.12 \
                   python3-pip \
                   python3.12-venv && \
    rm -rf /var/lib/apt/lists

WORKDIR /usr/src/

#must run from Project-WHIP
#whip2vec files
COPY figaro/src/ ./w2v/src/
COPY figaro/conf/ ./w2v/conf/
COPY figaro/requirements.txt .
COPY figaro/run.sh .

#life2vec files
COPY life2vec-light/src/ ./l2v/src/
COPY life2vec-light/conf/ ./l2v/conf/

# make sure to use venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        # --extra-index-url https://download.pytorch.org/whl/cu118 && \
        # --extra-index-url https://download.pytorch.org/whl/cpu && \
        --extra-index-url https://download.pytorch.org/whl/cu124 && \
    python3.12 -m pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
