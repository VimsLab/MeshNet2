# MeshNet++: A Network with a Face
Official PyTorch implementation for [MeshNet++: A Network with a Face](https://dl.acm.org/doi/abs/10.1145/3474085.3475468). The code has been implemented and tested on the Ubuntu operating system only.

## Install CUDA Toolkit and cuDNN
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the [cuDNN library](https://developer.nvidia.com/rdp/cudnn-archive) matching the version of your Ubuntu operating system. Installation of the Anaconda Python Distribution is required as well. We recommend installing CUDA10.1.


## Download data set
Download the pre-processed data sets in the datasets/<dataset>/raw/ directory. Files are in the OBJ file format (.obj). For the MSB and SHREC11 and all mesh model consists of precisely 500 faces. For ModelNet10, ModelNet40, and 3D-FUTURE all mesh model consists of precisely 1024 faces.

## Train
```
python train_val.py SHREC11 16-04_A
```

## Test
```
python test.py SHREC11 16-04_A
```
