# AdasNet

### Info
  Ningqi*， Chengxi Yang^， Wenxiu Sun^ and Binheng Song*. *Tsinghua University, ^SenseTime Group Limited.
  Emails: lnq16@mails.tsinghua.edu.cn
  reference “Unsupervised Binocular Depth Perception on Driving Scenario”
  submited to 24th International Conference on Pattern Recognition, ICPR'2018, [link](http://www.icpr2018.org/)
  
### Prerequisites
  - Tensorflow
  - OpenCv
  - Download our trained model through Baiduyun [link](https://pan.baidu.com/s/1c1GJzna)

### Testing on the KITTI dataset
  - take stereo images in ./input/left and ./input/right as input
  - run test.py
  - reslut(disparity, occlusion, warpedLeft) are show in ./output 

### Training
  - adjust loadData.py to fit dataset location
  - load trained model
  - run train.py
  - save models (checkpoint) in ./models
