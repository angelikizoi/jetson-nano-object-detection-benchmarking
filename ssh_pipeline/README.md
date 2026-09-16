# SSH-tunnelled remote-GPU offload pipeline

A hybrid edge/cloud setup: the Jetson Nano captures and displays video, but the
expensive YOLO inference runs on a remote workstation GPU. Frames go out, only a
small list of detections comes back. This lets a 2GB Nano show **high-accuracy**
detections (YOLO11x, 57M params) in **real time** — 11.8 FPS at 640×480, faster
than any model running on the Nano itself (see
[`../benchmarks/realtime_inference.csv`](../benchmarks/realtime_inference.csv)).

## Architecture

```
        JETSON NANO 2GB                         GPU WORKSTATION
  ┌───────────────────────────┐          ┌───────────────────────────┐
  │  camera → capture frame   │          │   server_inference.py     │
  │  cv2.imencode → JPEG       │  JPEG    │   YOLO11x on RTX 5060 Ti  │
  │  jetson_client.py ────────────────────►   model(frame)           │
  │                           │  frame   │                           │
  │  draw boxes ◄───────────────────────────  detections (pickled)   │
  │  cv2.imshow + FPS overlay │  dets    │                           │
  └───────────────────────────┘          └───────────────────────────┘
              └──────────────  SSH tunnel  ──────────────┘
                     ssh -N -L 5050:localhost:5050 user@GPU_SERVER
```

The Python uses **plain TCP sockets**. Security and NAT traversal are handled by
running that TCP connection through an **SSH local-port-forward**, so nothing is
exposed on the open network and no firewall holes are needed.

## Wire protocol

Every message in both directions is length-prefixed:

```
[ 4 bytes big-endian uint32 = payload length ][ payload ]
```

| Direction         | Payload                                                      |
|-------------------|-------------------------------------------------------------|
| Jetson → server   | JPEG-encoded frame bytes                                     |
| server → Jetson   | pickled `list` of `[x1, y1, x2, y2, conf, cls, label]`      |


## Running it

**1. Start the SSH tunnel from the Jetson:**

```bash
ssh -N -L 5050:localhost:5050 user@GPU_SERVER
```

**2. Start the server on the workstation:**

```bash
python server_inference.py --weights yolo11x.pt --host 0.0.0.0 --port 5050
```

**3. Start the client on the Jetson:**

```bash
python jetson_client.py --host localhost --port 5050 --width 640 --height 480
```

Press `q` in the display window to quit.

## Files

| File                   | Runs on      | Role                                              |
|------------------------|--------------|---------------------------------------------------|
| `jetson_client.py`     | Jetson Nano  | capture → encode → send → receive → draw → display |
| `server_inference.py`  | GPU server   | receive → decode → YOLO → return detections        |


