"""
Jetson-side client for the remote-GPU offload pipeline.

Captures frames from a camera on the Jetson Nano, JPEG-encodes and ships each
one to the remote inference server over a (tunnelled) TCP connection, receives
the detections back, draws the bounding boxes locally, and displays the result
with a live FPS overlay.

Because the Jetson only encodes/decodes and draws — the heavy inference runs on
the remote GPU — even a 2GB Nano can display high-accuracy detections in real
time. See the benchmark in ../benchmarks/realtime_inference.csv (YOLO11x over
the SSH offload reached 11.8 FPS at 640x480, faster than any on-device config).

The socket is plain TCP; run it through an SSH tunnel so nothing is exposed on
the network. On the Jetson:

    ssh -N -L 5050:localhost:5050 user@GPU_SERVER

then point this client at localhost:5050 (the default).

Wire protocol (must match server_inference.py):
    [4 bytes big-endian uint32 = payload length][payload]
  Jetson -> server : JPEG bytes (one frame)
  server -> Jetson : pickled list of [x1, y1, x2, y2, conf, cls, label]

Usage:
    python jetson_client.py --host localhost --port 5050 --width 640 --height 480
    python jetson_client.py --source 0 --jpeg-quality 80
    python jetson_client.py --no-display        # headless benchmarking (no GUI)
"""

import argparse
import pickle
import socket
import struct
import time

import cv2
import numpy as np


def recv_all(sock, n):
    """Receive exactly n bytes, or return None if the peer closed early."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_frame(sock, frame, jpeg_quality):
    """JPEG-encode a frame and send it length-prefixed. Returns encoded size."""
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    payload = encoded.tobytes()
    sock.sendall(struct.pack(">I", len(payload)) + payload)
    return len(payload)


def recv_detections(sock):
    """Receive one length-prefixed pickled detection list. None if closed."""
    raw_size = recv_all(sock, 4)
    if raw_size is None:
        return None
    size = struct.unpack(">I", raw_size)[0]
    payload = recv_all(sock, size)
    if payload is None:
        return None
    return pickle.loads(payload)


def draw_detections(frame, detections, conf_threshold):
    """Draw boxes and labels returned by the server onto the frame in place."""
    for x1, y1, x2, y2, conf, _cls, label in detections:
        if conf < conf_threshold:
            continue
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw, p1[1]), (0, 255, 0), -1)
        cv2.putText(frame, text, (p1[0], p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def open_capture(source, width, height):
    """Open a camera source. Integer -> device index; string -> path/pipeline."""
    try:
        source = int(source)
    except ValueError:
        pass  # keep as string (file path or GStreamer pipeline)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap


def parse_args():
    p = argparse.ArgumentParser(description="Jetson client for remote-GPU YOLO offload")
    p.add_argument("--host", default="localhost", help="Server host (localhost if SSH-tunnelled)")
    p.add_argument("--port", type=int, default=5050, help="Server port")
    p.add_argument("--source", default="0", help="Camera index, video path, or GStreamer pipeline")
    p.add_argument("--width", type=int, default=640, help="Capture/encode width")
    p.add_argument("--height", type=int, default=480, help="Capture/encode height")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100")
    p.add_argument("--conf-threshold", type=float, default=0.25, help="Min confidence to draw")
    p.add_argument("--no-display", action="store_true", help="Headless: skip the GUI window")
    return p.parse_args()


def main():
    args = parse_args()
    cap = open_capture(args.source, args.width, args.height)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to {args.host}:{args.port}. Press 'q' to quit.")

    # Exponential moving average of the end-to-end round-trip FPS.
    ema_fps = None
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            t0 = time.time()
            send_frame(sock, frame, args.jpeg_quality)
            detections = recv_detections(sock)
            if detections is None:
                print("Server closed the connection.")
                break
            dt = time.time() - t0

            inst_fps = 1.0 / dt if dt > 0 else 0.0
            ema_fps = inst_fps if ema_fps is None else 0.9 * ema_fps + 0.1 * inst_fps
            frames += 1

            draw_detections(frame, detections, args.conf_threshold)
            cv2.putText(frame, f"{ema_fps:4.1f} FPS  ({dt * 1000:.0f} ms round-trip)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if not args.no_display:
                cv2.imshow("Jetson remote-GPU detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frames % 30 == 0:
                print(f"{frames} frames | {ema_fps:.1f} FPS | {dt * 1000:.0f} ms round-trip")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        if ema_fps is not None:
            print(f"Stopped after {frames} frames. Mean throughput ~{ema_fps:.1f} FPS.")


if __name__ == "__main__":
    main()
