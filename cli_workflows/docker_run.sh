# Docker route: run the prebuilt Ultralytics image for JetPack 4 on the Jetson Nano.
# JetPack has native Docker support, so no CUDA setup is needed inside the container.


# Pull the Ultralytics image built for JetPack 4 (JetPack < 5.x uses the jetson-jetpack4 tag).
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4

# Run a container with:
#   --ipc=host        shared memory (PyTorch dataloaders / inference need it)
#   --runtime nvidia  GPU access
#   --device /dev/video0   pass the USB webcam through (for real-time inference)
sudo docker run -it --ipc=host --runtime nvidia \
  --device /dev/video0 \
  ultralytics/ultralytics:latest-jetson-jetpack4

# Then, inside the container, start python3 and drive YOLO from there:
#   from ultralytics import YOLO
#   model = YOLO("yolo26n.pt")          # nano model, NMS-free, ideal for edge
#   model.benchmark(data="coco128.yaml", device=0)      # benchmark in COCO128
#   model(source=0)                            # real-time inference on /dev/video0
#   model.export(format="onnx", data="coco128.yaml", device=0)       # ONNX export
