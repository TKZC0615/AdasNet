# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:18:52 2017

@author: sensetime
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from corr import corr1d
from warp import warpImage
from warp import warpDisparity

batch_size = 1
channel = 3
image_height = 448
image_width = 1472
image_width_long = 1550

lr = 0.0001


''' ========================================================= function ========================================================='''
'''===========================================
placeholder of input
    placeholder:
        1. img0: left image 
        2. img1: right image
        3.(removed) disp_gt_aug: ground truth 
    size:
        [batch_size, height, width, channel]
==========================================='''
def getPlaceholderInput():
    with tf.variable_scope('img0'):
        img0 = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))
    with tf.variable_scope('img1'):    
        img1 = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))
    with tf.variable_scope('img1long'):    
        img1long = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width_long, 3))
    with tf.variable_scope('dispR'):    
        dispR = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 1))
    return img0, img1, img1long, dispR


'''===========================================
get Conv2
    input:
        1. data: input data to be pad
        2. outnum: =
        3. kernelSize: [height, width]
        4. iPad: int; pad-size
        5. iStride: int; stride-size
        6. scopeStr: name of scope
    output:
        conv2D data (with customized padding)
==========================================='''
def getCustomizedConv2D(data, outnum, kernelSize, iPad, iStride, scopeStr='Convolution1', acti='relu'):
    with tf.variable_scope(scopeStr):
        data1 = tf.pad(data, [[0, 0], [iPad, iPad], [iPad, iPad], [0, 0]], "CONSTANT")
        if acti == 'sigmoid':
            data2 = slim.conv2d(data1, outnum, kernelSize, iStride, 'VALID', activation_fn=tf.nn.sigmoid)
        elif acti == 'elu':
            data2 = slim.conv2d(data1, outnum, kernelSize, iStride, 'VALID', activation_fn=tf.nn.elu)
        else:
            data2 = slim.conv2d(data1, outnum, kernelSize, iStride, 'VALID')
    return data2


'''===========================================
get Deconv (stride=2)
    input:
        1. data: input data to be pad
        2. outnum: =
        3. kernelSize: [height, width]
        4. iPad: int; pad-size
        5. scopeStr: name of scope
    output:
        deconv data (with customized padding and stride)
==========================================='''
def getCustomizedDeconv(data, outnum, kernelSize, iPad, scopeStr, acti='relu'):
    height = 2*data.get_shape().as_list()[1]-1+(kernelSize[0]-1-iPad)*2
    width = 2*data.get_shape().as_list()[2]-1+(kernelSize[1]-1-iPad)*2
    data = tf.image.resize_images(data, [height,width])
    if acti == 'sigmoid':
        data = slim.conv2d(data, outnum, kernelSize, 1, 'VALID', scope=scopeStr,activation_fn=tf.nn.sigmoid)
    elif acti == 'elu':
        data = slim.conv2d(data, outnum, kernelSize, 1, 'VALID', scope=scopeStr,activation_fn=tf.nn.elu)
    else:
        data = slim.conv2d(data, outnum, kernelSize, 1, 'VALID', scope=scopeStr)
    return data
    
    
'''===========================================
get shared Conv (conv1 and conv2 share the same convolution)
    input:
        1. i1: img1 or input 1
        2. i2: img2 or input 2
        3. scopeStr: name of scope
        4. kernelInfo: 4D list, [height, width, channel_in, channel_out]
        5. iPad: int; pad-size
        6. stridesList: 4D list, [batch, height, width, channel]
    output:
        convaa: ans of img1 through shared Conv
        convbb: ans of img2 through shared Conv
==========================================='''
def getSharedConv2D(i1, i2, scopeStr, kernelInfo, iPad, stridesList):
    with tf.variable_scope(scopeStr):
        kernel = tf.Variable(tf.truncated_normal(kernelInfo,dtype=tf.float32),name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[kernelInfo[3]],dtype=tf.float32),name='biases')
        with tf.variable_scope(scopeStr+'a'):
            i1 = tf.pad(i1, [[0, 0], [iPad, iPad], [iPad, iPad], [0, 0]], "CONSTANT")
            conva = tf.nn.conv2d(i1, kernel, stridesList, 'VALID')
            convaa = tf.nn.relu(tf.nn.bias_add(conva, biases))
        with tf.variable_scope(scopeStr+'b'):
            i2 = tf.pad(i2, [[0, 0], [iPad, iPad], [iPad, iPad], [0, 0]], "CONSTANT")
            convb = tf.nn.conv2d(i2, kernel, stridesList, 'VALID')
            convbb = tf.nn.relu(tf.nn.bias_add(convb, biases))
    return convaa,convbb,kernel,biases


'''===========================================
the repeat part of model
    input:
        1. img: = input 1; main flow
        2. conv1: = input 2; subflow1
        3. conv2: = input 3; subflow2
        4. outnum1: num of out in deconv of conv1
        5. outnum2: num of out in conv of concat
        6. nameDeconv: name of name_deconv
        7. nameFlow: name of name_flow
        8. nameConcat: name of name_concat
        9. nameConv1: name of name_conv1
        10. nameConv2: name of name_conv2
    work:
        action |  in           |      out         |     num_out  kernel_size  stride
        --------------------------------------------------------------------------
        deconv | conv1         -> deconvData      |    (outnum1,   [4,4],       2)
        deconv | conv2         -> upsample_flow   |    (1,         [4,4],       2)
               | img                              |
        concat | deconvData    -> concatData      |    axis=3
               | upsample_flow                    |
        conv   | concatData    -> C1              |    (outnum2,   [3,3],       1) 
        conv   | C1            -> C2              |    (1,         [3,3],       1) 
    output:
        C1: = subflow1
        C2: = subflow2
==========================================='''
def getRepeatPart(img, conv1, conv2, outnum1, outnum2, nameDeconv, nameFlow, nameConcat, nameConv1, nameConv2):
    with tf.variable_scope(nameDeconv):
        deconvData = getCustomizedDeconv(conv1, outnum1, [4, 4], 1, nameDeconv)
    with tf.variable_scope(nameFlow):
        upsample_flowxtoy = getCustomizedDeconv(conv2, 1, [4, 4], 1, nameFlow)
    with tf.variable_scope(nameConcat):
        myconcat = tf.concat([img, deconvData, upsample_flowxtoy], 3, name=nameConcat)
    with tf.variable_scope(nameConv1):
        C1 = getCustomizedConv2D(myconcat, outnum2, [3, 3], 1, 1, nameConv1)
    with tf.variable_scope(nameConv2):
        C2 = getCustomizedConv2D(C1, 1, [3, 3], 1, 1, nameConv2)
    return C1, C2


'''===========================================
calculate L1lossM
    input:
        1. targetImg: target image
        2. subDis: compared img
        3. weightLoss: weight of Loss
        4. scopeLoss: name of Loss
        5. scopeDownsample: name of Downsample
        6. Eltwise: 
            6.1 withEltwise: True or False
            6.2 fEltwise: float, weight of Eltwise
            6.3 scopeEltwise: name of Eltwise
    output:
        flow_loss
==========================================='''
def getL1Loss(Ilbar, imgL, weightLoss, scopeLossStr):
    with tf.variable_scope(scopeLossStr):
        with tf.variable_scope('flow_loss'):
            flow_loss = weightLoss * tf.reduce_mean(tf.abs(Ilbar - imgL))
    return flow_loss

def getGradientL1Loss(Ilbar, imgL, weightLoss, scopeLossStr):
    gxIlbar = Ilbar[:,:,:-1,:] - Ilbar[:,:,1:,:]
    gximgL = imgL[:,:,:-1,:] - imgL[:,:,1:,:]
    gyIlbar = Ilbar[:,:-1,:,:] - Ilbar[:,1:,:,:]
    gyimgL = imgL[:,:-1,:,:] - imgL[:,1:,:,:]
    with tf.variable_scope(scopeLossStr):
        with tf.variable_scope('flow_loss'):
            flow_loss = weightLoss * (tf.reduce_mean(tf.abs(gxIlbar - gximgL))+tf.reduce_mean(tf.abs(gyIlbar - gyimgL)))
    return flow_loss

'''===========================================
SSIM value
    SSIM [-1,1]
==========================================='''
def getSSIM(img0, img1, kernelSize):
    L = 1
    C1 = (0.01*L)**2
    C2 = (0.03*L)**2
    
    mu_x = slim.avg_pool2d(img0, kernelSize, 1, 'VALID')
    mu_y = slim.avg_pool2d(img1, kernelSize, 1, 'VALID')
    sigma_xx = slim.avg_pool2d(img0**2, kernelSize, 1, 'VALID')-mu_x**2
    sigma_yy = slim.avg_pool2d(img1**2, kernelSize, 1, 'VALID')-mu_y**2
    sigma_xy = slim.avg_pool2d(img0*img1, kernelSize, 1, 'VALID')-mu_x*mu_y
    
    ssim_1 = (2*mu_x*mu_y+C1)*(2*sigma_xy+C2)
    ssim_2 = (mu_x**2+mu_y**2+C1)*(sigma_xx+sigma_yy+C2)
    
    SSIM = ssim_1/ssim_2
    return SSIM

'''===========================================
SSIM Loss
    SSIM [-1,1]
    ssimLoss [0,1]
==========================================='''
def getSSIMLoss(Ilbar, imgL, weightLoss, scopeLossStr):
    kernelSize0 = 3
    ''' math with way of warp (percet or abs=value) '''
    with tf.variable_scope(scopeLossStr):
        with tf.variable_scope('ssim'):
            SSIM1 = getSSIM(imgL, Ilbar, kernelSize0)
        with tf.variable_scope('ssim_loss'):
            ssimLoss1 = tf.reduce_mean(tf.clip_by_value((1 - SSIM1) / 2.0, 0, 1))
        lossA = weightLoss * (ssimLoss1)
    return lossA


def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


'''======
Consider of smoothness with 8 points around x
        \ | /
        - x -
        / | \
======'''
def getDispSmoothLoss8(imgL, subDis, weightLoss, scopeLossStr):
    with tf.variable_scope(scopeLossStr):
        with tf.variable_scope('smooth_loss'):            
            disp_gradients_x = subDis[:,:,:-1,:] - subDis[:,:,1:,:]
            disp_gradients_y = subDis[:,:-1,:,:] - subDis[:,1:,:,:]
            # xie zhe de smooth
            disp_gradients_xy1 = subDis[:,:-1,:-1,:] - subDis[:,1:,1:,:]
            disp_gradients_xy2 = subDis[:,:-1,1:,:] - subDis[:,1:,:-1,:]
        
            image_gradients_x = imgL[:,:,:-1,:] - imgL[:,:,1:,:]
            image_gradients_y = imgL[:,:-1,:,:] - imgL[:,1:,:,:]
            # xie zhe de smooth
            image_gradients_xy1 = imgL[:,:-1,:-1,:] - imgL[:,1:,1:,:]
            image_gradients_xy2 = imgL[:,:-1,1:,:] - imgL[:,1:,:-1,:]
        
            image_gradients_x = 0.25*image_gradients_x**2
            image_gradients_y = 0.25*image_gradients_y**2
            image_gradients_xy1 = 0.25*image_gradients_xy1**2
            image_gradients_xy2 = 0.25*image_gradients_xy2**2
        
            weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
            weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))
            weights_xy1 = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_xy1), 3, keep_dims=True))
            weights_xy2 = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_xy2), 3, keep_dims=True))
        
            smoothness_x = disp_gradients_x**2 * weights_x
            smoothness_y = disp_gradients_y**2 * weights_y
            smoothness_xy1 = disp_gradients_xy1**2 * weights_xy1
            smoothness_xy2 = disp_gradients_xy2**2 * weights_xy2
            
            disp_left_loss  = tf.reduce_mean(tf.abs(smoothness_x))+tf.reduce_mean(tf.abs(smoothness_y)) \
                                + 1.0/np.sqrt(2)*tf.reduce_mean(tf.abs(smoothness_xy1)) \
                                + 1.0/np.sqrt(2)*tf.reduce_mean(tf.abs(smoothness_xy2))
    return weightLoss * disp_left_loss

def getOccSSIMLoss(Ilbar, imgL, subDis, weightLoss, scopeLossStr):
    with tf.variable_scope(scopeLossStr):

        ''' === Computing Occlusion Map === '''
        maxn = 30
        w = tf.shape(subDis)[2]
        offsetn = tf.zeros((tf.shape(subDis)[0], tf.shape(subDis)[1], maxn, tf.shape(subDis)[3]))*0.0
        subDistemp = tf.concat([subDis, offsetn], 2)
        downMapup = subDis * 0.0# + 1.0
        for n in range(1,maxn+1):
            subn = subDistemp[:,:,n:n+w,:]-subDis[:,:,:,:]-n
            downMapn = tf.sign(tf.sign(subn)-0.5)/2.0+0.5
            downMapdown = downMapup+downMapn-(downMapup * downMapn)
            downMapup = tf.sign(downMapdown)*1.0
        
        downMap = downMapdown
        upMap = -downMap+1.0
        
        ''' SSIM with Occlusion map '''
        
        C1 = (0.01)**2
        C2 = (0.03)**2
        # ssim init
        mu_x = slim.avg_pool2d(imgL, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(Ilbar, 3, 1, 'VALID')
        
        sigma_xx = slim.avg_pool2d(imgL**2, 3, 1, 'VALID')-mu_x**2
        sigma_yy = slim.avg_pool2d(Ilbar**2, 3, 1, 'VALID')-mu_y**2
        sigma_xy = slim.avg_pool2d(imgL*Ilbar, 3, 1, 'VALID')-mu_x*mu_y
        # ssim upmap
        upSmoothConfidence = slim.avg_pool2d(upMap, 3, 1, 'VALID')
        ssim_1 = (2*mu_x*mu_y+C1)*(2*sigma_xy+C2)
        ssim_2 = (mu_x**2+mu_y**2+C1)*(sigma_xx+sigma_yy+C2)
        
        upSSIM = ssim_1/ssim_2
        upSSIMloss = tf.clip_by_value((1 - upSSIM) / 2.0, 0, 1)
        upSSIMloss = upSmoothConfidence * upSSIMloss
        
        ssimLoss1 = tf.reduce_mean(upSSIMloss)
        # ssim downmap
        # avg ssim
        disaccount = 0.5
        avgssimloss = tf.reduce_sum(tf.reduce_sum(upSSIMloss)/(tf.reduce_sum(upSmoothConfidence)+0.0001))
        downSmoothConfidence = slim.avg_pool2d(downMap, 3, 1, 'VALID')
        downSSIMloss = downSmoothConfidence * avgssimloss * disaccount
        downSSIMloss = tf.clip_by_value(downSSIMloss, 0, 1)
        
        ssimLoss2 = tf.reduce_mean(downSSIMloss)
        # ssim all
        ssimLoss = ssimLoss1 + ssimLoss2

        return weightLoss * ssimLoss, downMap
        
def getLRconsistLoss(disR, subDis, weightLoss, scopeLossStr):
    disLbar = warpDisparity(disR, subDis/1472.0)
    with tf.variable_scope(scopeLossStr):
        with tf.variable_scope('flow_loss'):
            flow_loss = weightLoss * tf.reduce_mean(tf.clip_by_value(tf.abs(disLbar[:,:,120:1472-120,:] - subDis[:,:,120:1472-120,:]), 0.0, 1.0))
    return flow_loss, disLbar
    
''' ========================================================= model structure ========================================================='''


class AdasNet(object):
    def __init__(self):
        '''model'''
        ''' part 1: input and  corr '''
        self.img0, self.img1, self.img1long, self.dispR = getPlaceholderInput()
        # Eltwise 1/255
        with tf.variable_scope('img0_aug'):
            img0_aug = tf.multiply(1/255.0, self.img0)
        with tf.variable_scope('img1_aug'):
            img1_aug = tf.multiply(1/255.0, self.img1)
        # share conv2D
        conv1a,conv1b,self.conv1_w,self.conv1_b = getSharedConv2D(img0_aug, img1_aug, 'conv1', [7,7,3,64], 3, [1,2,2,1])
        conv2a,conv2b,self.conv2_w,self.conv2_b = getSharedConv2D(conv1a, conv1b, 'conv2', [5,5,64,128], 2, [1,2,2,1])
        # correlation
        with tf.variable_scope('corr'):
            corr = corr1d(conv2a,conv2b)
        # concat / flow start
        conv_redir = slim.conv2d(conv2a, 64, [1, 1], 1, 'VALID', scope='conv_redir')#CUDNN
        concat2 = tf.concat([corr, conv_redir], 3, name='concat2')
        
        ''' part 2: feature beyond Convolution '''
        # convData = getCustomConv2D( input, numOut, kernelSize, pad, stride, scope)
        conv3 = getCustomizedConv2D(concat2, 256,  [5, 5], 2, 2, 'conv3')
        conv3_1 = getCustomizedConv2D(conv3, 256,  [3, 3], 1, 1, 'conv3_1')
        conv4 = getCustomizedConv2D(conv3_1, 512,  [3, 3], 1, 2, 'conv4')
        conv4_1 = getCustomizedConv2D(conv4, 512,  [3, 3], 1, 1, 'conv4_1')
        conv5 = getCustomizedConv2D(conv4_1, 512,  [3, 3], 1, 2, 'conv5')
        conv5_1 = getCustomizedConv2D(conv5, 512,  [3, 3], 1, 1, 'conv5_1')
        conv6 = getCustomizedConv2D(conv5_1, 1024, [3, 3], 1, 2, 'conv6')
        conv6_1 = getCustomizedConv2D(conv6, 1024, [3, 3], 1, 1, 'conv6_1')
        Convolution1 = getCustomizedConv2D(conv6_1, 1, [3, 3], 1, 1, 'Convolution1')
        
        ''' part 3: Flow '''
        #  subflow1,   subflow2     = getRepeatPart(img,     conv1,        conv2, outnum1, outnum2, nameDeconv, nameFlow, nameConcat, nameConv1, nameConv2)
        Convolution2, Convolution3  = getRepeatPart(conv5_1, conv6_1,      Convolution1, 512,512,'deconv5','upsample_flow6to5','concat3','Convolution2','Convolution3')
        Convolution4, Convolution5  = getRepeatPart(conv4_1, Convolution2, Convolution3, 256,256,'deconv4','upsample_flow5to4','concat4','Convolution4','Convolution5')
        Convolution6, Convolution7  = getRepeatPart(conv3_1, Convolution4, Convolution5, 128,128,'deconv3','upsample_flow4to3','concat5','Convolution6','Convolution7')
        Convolution8, Convolution9  = getRepeatPart(conv2a,  Convolution6, Convolution7,  64, 64,'deconv2','upsample_flow3to2','concat6','Convolution8','Convolution9')
        Convolution10,Convolution11 = getRepeatPart(conv1a,  Convolution8, Convolution9,  32, 32,'deconv1','upsample_flow2to1','concat7','Convolution10','Convolution11')
        Convolution12,Convolution13 = getRepeatPart(img0_aug,Convolution10,Convolution11, 16, 32,'deconv0','upsample_flow1to0','concat8','Convolution12','Convolution13')
        
        ''' part 4: Loss '''
        #  getSSIMLoss( targetImg, subDis, weightLoss, scopeLossStr, scopeDownsampleStr, withEltwise=False, fEltwise=1.0, scopeEltwiseStr='Eltwise_d'):    
        self.Ilbar = warpImage(self.img1long, Convolution13/1550.0)
        # L1
        self.flow_loss0_l1 = getL1Loss(self.Ilbar, self.img0, 32,   'flow_loss0_l1')
        # L1
        self.flow_loss0_gl1 = getGradientL1Loss(self.Ilbar, self.img0, 32,   'flow_loss0_gl1')
        # SSIM
        self.flow_loss0_ssim = getSSIMLoss(self.Ilbar, self.img0, 32,   'flow_loss0_ssim')
        # smooth Disp
        self.flow_loss0_smooth = getDispSmoothLoss8(self.img0, Convolution13, 32,   'flow_loss0_smooth')
        # confidence loss
        self.flow_loss0_nocssim,self.occ =  getOccSSIMLoss(self.Ilbar, self.img0, Convolution13, 32, 'flow_loss0_confidence')
        # LR consist Loss
        self.flow_loss0_consist,self.dispLbar = getLRconsistLoss(self.dispR, Convolution13, 32, 'flow_loss0_consist')
        # sum
#        self.flow_loss0 = 0.2*(0.15*self.flow_loss0_l1+0.85*self.flow_loss0_ssim)+0.2*self.flow_loss0_smooth+1.0*self.flow_loss0_confidence
        self.flow_loss0 =   0.00*self.flow_loss0_ssim \
                          + 0.10*self.flow_loss0_smooth \
                          + 0.01*self.flow_loss0_gl1 \
                          + 0.01*self.flow_loss0_l1 \
                          + 0.90*self.flow_loss0_nocssim \
                          + 0.01*self.flow_loss0_consist
        # all
        ''' we will train in all 6 sub disp in the very beginning'''
        #self.lossSum = self.flow_loss6 + self.flow_loss5 + self.flow_loss4 + self.flow_loss3 + self.flow_loss2 + self.flow_loss1 + self.flow_loss0
        self.lossSum = self.flow_loss0
        ''' part 5: train operation '''
        self.train = tf.train.AdamOptimizer(lr).minimize(self.lossSum)
        
        ''' potential disparity '''
        self.disparity = Convolution13 #* 1000.0



