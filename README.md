# AdasNet

## Info
**Unsupervised Binocular Depth Perception on Driving Scenario**<br>
Ningqi Luo', Chengxi Yang", Wenxiu Sun" and Binheng Song'.<br>
'Tsinghua University, "SenseTime Group Limited.<br>
Email: lnq16@mails.tsinghua.edu.cn<br>
Submitted to 24th International Conference on Pattern Recognition, [ICPR'2018](http://www.icpr2018.org/)

## Introduction
To achive depth perception on drving scenario (i) We construct a CNN-based neural network which takes pairwise stereo images as input and output disparity map of the single left image. The training process of this neural network is not reliant on ground truth disparities. (ii) We propose an occlusion-aware reconstruction loss function. Nowadays the similarity evaluations
between reconstructed and reference images do not consider of the occlusion area. We design an occlusion module and add it
into matching loss. (iii) We train our unsupervised model on KITTI Stereo 2012, Stereo 2015, Cityscapes and our self-collected dataset. Trained model is available in [Baiduyun](https://pan.baidu.com/s/1c1GJzna).

## How to Use
### Prerequisites
  - Tensorflow
  - OpenCv
  - Download our trained model from [Baiduyun](https://pan.baidu.com/s/1c1GJzna)

### Testing
  - Take stereo images in ./input/left and ./input/right as input
  - Run test.py
  - Reslut(disparity, occlusion, warpedLeft) are show in ./output 

### Training
  - Adjust loadData.py to fit dataset location
  - Load trained model
  - Run train.py
  - Save models (checkpoint) in ./models

## Related work
C. Godard, O. Mac Aodha, and G. J. Brostow, “Unsupervised monocular depth estimation with left-right consistency,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, pages 279–270, 2016. [Github](https://github.com/mrharicot/monodepth)<br>
Y. Zhong, Y. Dai, and H. Li, “Self-supervised learning for stereo matching with self-improcing ability,” in arXiv preprint arXiv: 1709.00930, 2017.

## Acknowledgment
We gratefully acknowledge the support from department of data acquisition in SenseTime Group Limited for providing driving dataset. This work is supported by Tsinghua-Sensetime practice base.
