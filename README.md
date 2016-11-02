# dense-trajectory-gpu
This repository contains the code for computing the dense trajectory for any video given as input. In this code the CPU version of farneback optical flow used in the original paper is replaced by GPU version of it so that the overall computation time for dense trajectories is greatly reduced.

### Prerequisites
* CUDA capable GPU
* OpenCV with CUDA support
* FFMPEG

### Compilation
* The compilation of the code can be done by following steps: 
```

    git clone https://github.com/AadityaVora/dense-trajectory-gpu.git
```
