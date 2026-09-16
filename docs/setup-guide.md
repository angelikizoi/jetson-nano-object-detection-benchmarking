# Setup guide — object detection on JetPack 4.6.1 (Jetson Nano 2GB)

The Jetson Nano 2GB ships with **JetPack 4.6.1**: Ubuntu 18.04, **Python 3.6.9**,
CUDA 10.2, OpenCV 4.1.1 (no CUDA), Docker 18.09.2, TensorRT 8.2.1. That Python
3.6 / ARM64 combination is the source of nearly every difficulty below —
Ultralytics requires Python ≥ 3.8, and the usual PyTorch wheels don't target
this platform.

This guide documents both routes we tried: the **Docker** route (easy, but
higher latency) and the **native** route (harder, but faster and more flexible).


## Monitoring

Install `jetson-stats` to watch CPU/GPU/RAM utilisation and component
temperatures while benchmarking:

```bash
sudo pip3 install jetson-stats
jtop
```

## Route A — Docker (recommended for a quick start)

JetPack has native Docker support, so for JetPack < 5.x Ultralytics provides a
prebuilt image. Pull it and run a container with GPU access, shared memory, and
(for the webcam) the video device:

See [`../cli_workflows/docker_run.sh`](../cli_workflows/docker_run.sh) for the
exact `docker run` invocation. Inside the container you can immediately load a
model and validate — the library fetches weights and COCO128 on first use.

**Known limitation:** we could not export to TensorRT inside the container,
likely an onnx/TensorRT version mismatch between the container and the host.
Real-time webcam display in Docker also needs an ARM64 OpenCV wheel plus the
right `--device`/display flags.

## Route B — Native install (harder, faster)

You **cannot** `pip install ultralytics` here — it needs Python ≥ 3.8. Rather
than fight to install a newer Python (which then needs custom-built ARM64 wheels
for torch/torchvision/onnxruntime-gpu), we installed the dependencies manually
and ran detection from a YOLO source repository.

We used the **YOLOv9** reference implementation
([WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)) rather than the full
Ultralytics repo, because its source is smaller and single-purpose.

### Workflow

| #  | Step |
|----|------|
| 1  | System update |
| 2  | Verify 5GB **SWAP** was configured during initial setup (needed on 2GB) |
| 3  | Install `pip` |
| 4  | Install `numpy` |
| 5  | Rebuild **OpenCV 4.5.1 with CUDA** ([build walkthrough](https://www.youtube.com/watch?v=P-EZr0zy53g)) |
| 6  | Install **torch + torchvision** wheels for JetPack 4.6.1 ([NVIDIA forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)) |
| 7  | `pip install` pandas, psutil, seaborn, matplotlib, ipython, tqdm, requests |
| 8  | `git clone https://github.com/WongKinYiu/yolov9.git` |
| 9  | Download model weights (e.g. `yolov9-c`, `yolov9-t`) |
| 10 | Run validation & detection from the repo CLI (see [`validation_detection.md`](../cli_workflows/validation_detection.md)) |

### TensorRT export (the hardest part)

Export is memory- and compute-intensive — treat it like training, run it once
before deployment. The intermediate path is **PyTorch `.pt` → ONNX → TensorRT**.

| #  | Step |
|----|------|
| 11 | Install `protobuf` **3.0.0** as a system package |
| 12 | Install `onnxruntime-gpu` for JetPack 4.6.1 ([Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime)) |
| 13 | Install `onnx` **1.11.1** with `--no-deps` to avoid protobuf conflicts |

**Constraints we hit:**
- The YOLOv9 reference repo hadn't been updated in ~2 years, causing export
  incompatibilities on modern desktop environments that we couldn't resolve.
- On the Nano, export only succeeded for the **smallest** model (YOLOv9-T, ~3.2M
  params) — the 2GB memory ceiling blocked larger ones.

## Result

Once installed, YOLOv9-T exported to a TensorRT engine ran real-time webcam
inference at ~10.8 FPS (640×480) natively — see
[`../benchmarks/realtime_inference.csv`](../benchmarks/realtime_inference.csv).
