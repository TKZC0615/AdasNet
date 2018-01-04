# AdasNet

### Prerequisites
  - Tensorflow
  - OpenCv
  - Download our trained model through Baiduyun [link](https://pan.baidu.com/s/1c1GJzna)

### Testing on the KITTI dataset
  - take stereo images in ./input/left and ./input/right as input
  - run test.py
  - reslut(disparity, occlusion, warpedLeft) are show in ./output 

### Training
  We do not provide the code for training. For training, we need an in-house differentiable *interpolation* layer developed by our company, [SenseTime Group Limited](https://www.sensetime.com/).
