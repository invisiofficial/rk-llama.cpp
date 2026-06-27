# Containerized RKNPU2 build of llama.cpp — Rockchip NPU GGML backend.
#
# Builds the in-tree source with the RKNPU2 backend and ships llama-server together with the
# vendored Rockchip runtime (librknnrt.so). Target: arm64 / RK3588(S) (e.g. Orange Pi 5 / 5 Pro).
#
# The NPU is reached from the container via the host DRM render node + dma-heap, so run with the
# devices mapped in, e.g.:
#   docker run --rm -p 8080:8080 \
#     --device /dev/dri --device /dev/dma_heap \
#     -e RKNPU_HYBRID=W8A8_STANDARD --ulimit nofile=65536 \
#     -v /path/to/models:/models \
#     <image> -m /models/model-Q8_0.gguf --host 0.0.0.0 --port 8080 -np 1 --jinja
# See docs/rknpu2-docker.md for the full run guide (quantization, tool-calling, caveats).

ARG UBUNTU_VERSION=bookworm

FROM debian:${UBUNTU_VERSION} AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_RKNPU2=ON \
        -DLLAMA_CURL=ON \
    && cmake --build build -j"$(nproc)" --target llama-server llama-cli llama-bench

FROM debian:${UBUNTU_VERSION}-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Built binaries + the ggml/llama shared libraries (incl. libggml-rknpu2.so).
COPY --from=build /src/build/bin/ /opt/llama/bin/
# Vendored Rockchip NPU userspace runtime — libggml-rknpu2.so dlopens/links against it at runtime.
COPY --from=build /src/ggml/src/ggml-rknpu2/libs/librknnrt.so /usr/lib/librknnrt.so
RUN ldconfig
ENV LD_LIBRARY_PATH=/opt/llama/bin
WORKDIR /opt/llama
EXPOSE 8080
ENTRYPOINT ["/opt/llama/bin/llama-server"]
CMD ["--help"]
