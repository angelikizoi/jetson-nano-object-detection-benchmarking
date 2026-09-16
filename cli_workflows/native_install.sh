# Native install route on JetPack 4.6.1 (Python 3.6, CUDA 10.2, ARM64).


# 1. System update
sudo apt-get update && sudo apt-get upgrade -y

# 2. Verify 5GB swap exists (essential on a 2GB board). Configure in initial setup.
free -h
swapon --show

# 3. pip
sudo apt-get install -y python3-pip
python3 -m pip install --upgrade pip

# 4. numpy
python3 -m pip install "numpy"

# 5. OpenCV 4.5.1 WITH CUDA — build from source (removes the stock no-CUDA 4.1.1).
#    Walkthrough: https://www.youtube.com/watch?v=P-EZr0zy53g&list=PLv8Cp2NvcY8AkXRldCAYCvFxRUs0h5JJF&index=4

## 5.1 Required dependencies
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
sudo ldconfig
sudo apt-get install build-essential cmake git unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgtk2.0-dev libcanberra-gtk*
sudo apt-get install python3-dev python3-numpy python3-pip
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install libv4l-dev v4l-utils
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libavresample-dev libvorbis-dev libxine2-dev
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install liblapack-dev libeigen3-dev gfortran
sudo apt-get install libhdf5-dev protobuf-compiler
sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev

## 5.2 Download OpenCV
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip 
unzip opencv.zip 
unzip opencv_contrib.zip

## 5.3 Rename dirs - build OpenCV
mv opencv-4.5.1 opencv
mv opencv_contrib-4.5.1 opencv_contrib
rm opencv.zip
rm opencv_contrib.zip

cd ~/opencv
mkdir build
cd build 

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 -D WITH_OPENCL=OFF -D WITH_CUDA=ON -D CUDA_ARCH_BIN=5.3 -D CUDA_ARCH_PTX="" -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_NEON=ON -D WITH_QT=OFF -D WITH_OPENMP=ON -D WITH_OPENGL=ON -D BUILD_TIFF=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_TBB=ON -D BUILD_TBB=ON -D BUILD_TESTS=OFF -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_LIBV4L=ON -D OPENCV_ENABLE_NONFREE=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=TRUE -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF ..
make -j4

cd ~
sudo rm -r /usr/include/opencv4/opencv2
cd ~/opencv/build
sudo make install
sudo ldconfig
make clean
sudo apt-get update 



# 6. torch + torchvision wheels for JetPack 4.6.1 (ARM64, Python 3.6).
#    Source: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

## 6.1 PyTorch
# TODO: download the torch wheel v1.10.0 from the NVIDIA forum https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl (wget did not work)
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl

## 6.2 torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.0 https://github.com/pytorch/vision
cd torchvision
export BUILD_VERSION=0.11.0  
python3 setup.py install --user
cd ../  # attempting to load torchvision from build dir will result in import error


# 7. Remaining Python deps
python3 -m pip install pandas psutil seaborn matplotlib ipython tqdm requests

# 8. Clone the YOLOv9 reference repo (smaller/single-purpose than full Ultralytics)
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9

# 9. Download weights (examples)
#    Grab yolov9-c (accuracy) and yolov9-t (edge) from the repo's releases.
# TODO: wget the weight .pt files from the WongKinYiu/yolov9 releases page.

# 10. Validation & detection are run from the repo CLI — see validation_detection.md

# ---- TensorRT export path (.pt -> ONNX -> TensorRT) ----

# 11. protobuf 3.0.0 as a system package
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# 12. onnxruntime-gpu for JetPack 4.6.1
#    Source: https://elinux.org/Jetson_Zoo#ONNX_Runtime
wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl


# 13. onnx 1.11.1 WITHOUT deps (avoids protobuf version conflicts)
python3 -m pip install onnx==1.11.1 --no-deps


