# Deployment comparison

Three ways to run a detector for the Jetson, and when each makes sense.

## 1. Docker

**Pros:** trivial to start — pull the prebuilt Ultralytics image for JetPack 4,
run a container, and any model works out of the box (weights and datasets
auto-download). No dependency hell.

**Cons:** measurable latency overhead on a memory-constrained device. On the 2GB
Nano, the *same* YOLOv9-C at 640×640 took **982 ms/frame in Docker vs 588 ms
native** — a consequence of container memory overhead on a shared-memory board.
We also could not export to TensorRT inside the container (onnx/TensorRT version
mismatch vs the host), and real-time webcam display needed extra ARM64 OpenCV
and device-flag work.

**Use when:** you want to try models quickly, or latency isn't critical.

## 2. Native install

**Pros:** lower latency, full control, and the flexibility to export to a
TensorRT engine — which gave the biggest on-device speedup we measured.

**Cons:** the setup is genuinely hard on JetPack 4.6.1 (Python 3.6, ARM64
wheels). You install torch/torchvision/onnxruntime-gpu from JetPack-specific
wheels, rebuild OpenCV with CUDA, and resolve protobuf/onnx conflicts by hand.
See [`setup-guide.md`](setup-guide.md).

**Use when:** you're deploying for real and want the best on-device performance.

## 3. Remote-GPU offload (SSH tunnel)

**Pros:** the Nano only captures, encodes, and draws — inference runs on a remote
GPU. This let a **57M-parameter YOLO11x** run at **11.8 FPS** on the Nano's
display, *faster and more accurate* than anything the Nano could run locally. The
SSH tunnel keeps it private with no open ports.

**Cons:** adds a hard network dependency and a round-trip latency floor; useless
if connectivity drops. Not a true "edge" solution — it's edge capture + cloud
compute.

**Use when:** the device has a reliable link to a GPU server and you need
accuracy the edge hardware can't deliver alone. Code: [`../ssh_pipeline/`](../ssh_pipeline/).

## Summary

| Approach        | Setup effort | On-device latency | Max model | Network needed |
|-----------------|--------------|-------------------|-----------|----------------|
| Docker          | Low          | Highest           | any       | no             |
| Native          | High         | Low               | small–mid | no             |
| Remote offload  | Medium       | Lowest (w/ link)  | any       | yes            |
