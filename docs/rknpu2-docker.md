# Running the RKNPU2 backend in a container

This guide covers building and running the Rockchip NPU (RKNPU2) backend of llama.cpp as a container on
RK3588(S) boards (Orange Pi 5 / 5 Pro, Radxa Rock 5, etc.). The Dockerfile is
[`.devops/rknpu2.Dockerfile`](../.devops/rknpu2.Dockerfile).

## Requirements (host)

- An RK3588(S) board on a **Rockchip BSP / vendor kernel** (e.g. Armbian `*-vendor-rk35xx`, kernel 6.1).
  The mainline kernel does **not** expose the `rknpu` driver that the userspace runtime binds to.
- RKNPU kernel driver **≥ 0.9.6** (check: `cat /sys/kernel/debug/rknpu/version`). Avoid 0.9.7.
- The NPU exposed as a DRM render node plus the dma-heap devices. On most boards the NPU is
  `/dev/dri/renderD129` and the GPU is `renderD128`, **but the numbering is not stable across boots /
  kernels** — map the whole `/dev/dri` directory rather than a fixed node.

The container bundles the matching `librknnrt.so` (vendored in the repo), so the host does not need it.

## Build

```sh
docker build -f .devops/rknpu2.Dockerfile -t rk-llama-server .
```

The image is arm64-native; build it on the board (or any arm64 builder) — no cross-compile/QEMU needed.

## Run

```sh
docker run --rm -p 8080:8080 \
  --device /dev/dri --device /dev/dma_heap \
  -e RKNPU_HYBRID=W8A8_STANDARD \
  --ulimit nofile=65536 \
  -v "$PWD/models:/models" \
  rk-llama-server \
  -m /models/your-model-Q8_0.gguf --host 0.0.0.0 --port 8080 -np 1 --jinja
```

Then hit the OpenAI-compatible API at `http://localhost:8080/v1`.

If your runtime/distro restricts device access, `--privileged` (instead of the `--device` flags) is the
simplest fallback for a trusted host.

## Notes & recommendations

- **Use `Q8_0` GGUF.** The backend requantizes GGUF weights into the NPU's native formats at load; the
  `Q8_0 → W8A8` path is the fast one. `Q4_0` (→ W4A4-Hadamard) is noticeably slower on the NPU.
- **The NPU's win is prompt processing (prefill / time-to-first-token)** — typically several times faster
  than CPU. Token generation is roughly on par with CPU. Long prompts benefit most.
- **One NPU consumer per device.** Running two processes that touch the NPU simultaneously can crash the
  inference process (and, on some driver versions, the system). Keep a single `llama-server` per board and
  serialize (`-np 1`).
- **`--jinja`** uses the model's embedded chat template, which is required for correct tool/function
  calling. Models with a dedicated tool-call format in llama.cpp (e.g. Gemma) return structured
  `tool_calls`.
- For best/steadiest throughput, pin the NPU devfreq governor to `performance`:
  `echo performance | sudo tee /sys/class/devfreq/*npu*/governor`.
- `RKNPU_DEVICE` selects the SoC (default `RK3588`); `RKNPU_CORES` restricts cores (default = all 3).

## Kubernetes

Map the whole `/dev/dri` directory and `/dev/dma_heap` via `hostPath`, run privileged, and pin one pod
per node:

```yaml
spec:
  nodeName: <board-node>
  containers:
    - name: rk-llama-server
      image: rk-llama-server
      args: ["-m", "/models/model-Q8_0.gguf", "--host", "0.0.0.0", "--port", "8080", "-np", "1", "--jinja"]
      env:
        - { name: RKNPU_HYBRID, value: W8A8_STANDARD }
      securityContext: { privileged: true }
      volumeMounts:
        - { name: dri,      mountPath: /dev/dri }
        - { name: dma-heap, mountPath: /dev/dma_heap }
        - { name: models,   mountPath: /models }
  volumes:
    - { name: dri,      hostPath: { path: /dev/dri,      type: Directory } }
    - { name: dma-heap, hostPath: { path: /dev/dma_heap, type: Directory } }
    - { name: models,   persistentVolumeClaim: { claimName: models } }
```
