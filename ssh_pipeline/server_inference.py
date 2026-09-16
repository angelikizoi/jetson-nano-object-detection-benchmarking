"""
Remote GPU inference server for the Jetson offload pipeline.

Runs on the workstation with the GPU. Listens on a TCP port, receives
JPEG-encoded frames from the Jetson (each prefixed with a 4-byte big-endian
length), runs YOLO inference, and returns the detections as a length-prefixed
pickled list.

The connection is expected to be tunnelled over SSH (see README.md), so the
socket itself is plain TCP bound to localhost on the far side of the tunnel.

Wire protocol (both directions):
    [4 bytes big-endian uint32 = payload length][payload]
  Jetson -> server payload : JPEG bytes (one frame)
  server -> Jetson payload : pickled list of detections, each
                             [x1, y1, x2, y2, conf, cls, label]

Usage:
    python server_inference.py --weights yolo11x.pt --host 0.0.0.0 --port 5050
"""

import argparse
import pickle
import socket
import struct

import cv2
import numpy as np
from ultralytics import YOLO


def recv_all(conn, n):
    """Receive exactly n bytes, or return None if the peer closed early."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Remote GPU YOLO inference server")
    p.add_argument("--weights", default="yolo11x.pt", help="YOLO weights to load")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=5050, help="Bind port")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        s.listen(1)
        print(f"Waiting for Jetson on {args.host}:{args.port} ...")
        conn, addr = s.accept()
        print(f"Connected: {addr}")

        with conn:
            while True:
                raw_size = recv_all(conn, 4)
                if not raw_size:
                    break
                size = struct.unpack(">I", raw_size)[0]

                raw_frame = recv_all(conn, size)
                if raw_frame is None:
                    break

                frame = cv2.imdecode(
                    np.frombuffer(raw_frame, np.uint8),
                    cv2.IMREAD_COLOR,
                )

                results = model(frame, verbose=False)[0]

                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    label = model.names[cls]
                    detections.append([x1, y1, x2, y2, conf, cls, label])

                payload = pickle.dumps(detections)
                conn.sendall(struct.pack(">I", len(payload)) + payload)

    print("Connection closed.")


if __name__ == "__main__":
    main()
