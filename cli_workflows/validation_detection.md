# Validation & detection — CLI commands

The offline-validation and detection benchmarks were produced by running the
[WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) repo scripts from the
terminal. This file records the command shapes used.

Download manually the model yolov9-t from https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt


## Offline validation (COCO128)

Validation reports mAP@0.5, mAP@0.5:0.95, and per-stage timing (pre-process,
inference, NMS) — the numbers in
[`../benchmarks/offline_validation_yolov9c.csv`](../benchmarks/offline_validation_yolov9c.csv).

```bash
# from inside the yolov9/ checkout
python val.py \
  --data data/coco128.yaml \
  --weights yolov9-t-converted.pt \
  --img 640 \
  --device 0            
```

Re-run with `--img 416` for the lower-resolution rows, and swap `--weights` for
the exported engine.

## Detection / inference

```bash
python detect.py \
  --weights yolov9-t-converted.pt \
  --source 0 \
  --img 640 \
  --device 0
```

## TensorRT export (.pt → engine)

```bash
python export.py --weights yolov9-t-converted.pt --include engine --img 640 --data data/coco128.yaml
```

On the Nano this succeeded only for the smallest model (yolov9-t) — larger
variants exceeded the memory ceiling during export.
