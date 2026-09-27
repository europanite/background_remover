FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        sudo; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt /root/requirements.txt
RUN set -eux; \
    python -m pip install --upgrade pip setuptools wheel; \
    pip install --no-cache-dir -r /root/requirements.txt

ARG HOST_USERNAME=user
ARG HOST_GROUPNAME=group
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV HOST_USERNAME=${HOST_USERNAME}
ENV HOST_GROUPNAME=${HOST_GROUPNAME}
ENV HOST_UID=${HOST_UID}
ENV HOST_GID=${HOST_GID}

RUN groupadd -g "$HOST_GID" "$HOST_GROUPNAME" && \
    useradd -m -s /bin/bash -u "$HOST_UID" -g "$HOST_GID" "$HOST_USERNAME" && \
    usermod -aG sudo "$HOST_USERNAME" && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /app
COPY . /app

USER $HOST_USERNAME
CMD ["python", "/app/main.py", "--help"]
