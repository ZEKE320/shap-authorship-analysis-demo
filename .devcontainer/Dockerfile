FROM mcr.microsoft.com/devcontainers/python:3.11

RUN --mount=type=cache,target=/var/lib/apt/lists \
  --mount=type=cache,target=/var/cache/apt/archives \
  apt update &&\
  apt upgrade -y
