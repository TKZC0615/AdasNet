# AdasNet

## Info
  **Unsupervised Binocular Depth Perception on Driving Scenario**
  Ningqi*, Chengxi Yang', Wenxiu Sun' and Binheng Song*.<br>
  *Tsinghua University, 'SenseTime Group Limited.<br>
  Emails: lnq16@mails.tsinghua.edu.cn<br>
  Submited to 24th International Conference on Pattern Recognition, [ICPR'2018](http://www.icpr2018.org/)

## Introduction
To achive depth perception on drving scenario (i) We construct a cascade residual network which takes pairwise stereo images as input and output disparity map of the single left image. The training process of this neural network is not reliant on ground truth disparities. (ii) We propose an occlusion-aware reconstruction matching loss function. Nowadays the similarity evaluations
between reconstructed and conference images do not consider of the occlusion area. We design an occlusion module and add it
into matching loss. (iii) We train our unsupervised model on KITTI stereo 2012, stereo 2015, Cityscapes and our own dataset. Trained model is available in [Baiduyun](https://pan.baidu.com/s/1c1GJzna).

## Use it
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

## Related work
C. Godard, O. Mac Aodha, and G. J. Brostow, “Unsupervised monocular depth estimation with left-right consistency,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, pages 279–270, 2016.<br>
[Github](https://github.com/mrharicot/monodepth)<br>
Y. Zhong, Y. Dai, and H. Li, “Self-supervised learning for stereo matching with self-improcing ability,” in arXiv preprint arXiv: 1709.00930, 2017.

## Acknowledgment
We gratefully acknowledge the support from department of data acquisition in SenseTime Group Limited for providing driving dataset. This work is supported by Tsinghua-Sensetime practice base.
