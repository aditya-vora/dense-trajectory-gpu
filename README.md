# dense-trajectory-gpu
This repository contains the code for computing the dense trajectory for any video given as input. In this code the CPU version of farneback optical flow used in the original paper is replaced by GPU version of it so that the overall computation time for dense trajectories is greatly reduced.

### Prerequisites
* CUDA capable GPU
* OpenCV with CUDA support
* FFMPEG

### Compilation and running the code
* The compilation of the code can be done by following steps: 
```
git clone https://github.com/AadityaVora/dense-trajectory-gpu.git
cd dense-trajectory-gpu
g++ DenseTrack.cpp -o DenseTrack.o `pkg-config --cflags --libs opencv` -lopencv_gpu
./DenseTrack.o ./test_sequences/<video-name>
```
### Referances

* [1] Wang, Heng, et al. "Action recognition by dense trajectories." Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.
