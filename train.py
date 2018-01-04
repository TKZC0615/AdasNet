#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:32:51 2017

@author: sensetime
"""

import tensorflow as tf
from scipy import misc
import AdasNet
import loadData
import numpy as np

def trainAdas():
    # dataset :
    #   1. Adas:       sensetimeData
    #   2. KITTI2012:  kitti2012
    #   3. KITTI2015:  kitti
    #   4. Cityscapes: cityscape  
    imageList = loadData.getImagesList('sensetimeData')
    volumeImage = len(imageList[0])
    
    with tf.Session() as sess:
        
        adas = AdasNet.AdasNet()
        
        new_saver = tf.train.Saver()
        new_saver.restore(sess, './models/adasNet-122500')
        
        tf.summary.scalar('lossSum', adas.lossSum)
        tf.summary.scalar('lossgl1', adas.flow_loss0_gl1)
        tf.summary.scalar('lossl1', adas.flow_loss0_l1)
        tf.summary.scalar('lossSSIM', adas.flow_loss0_ssim)
        tf.summary.scalar('lossSmooth', adas.flow_loss0_smooth)
        tf.summary.scalar('lossNocSSIM', adas.flow_loss0_nocssim)
        tf.summary.scalar('lossConsist', adas.flow_loss0_consist)
        
        tf.summary.image('disparity', adas.disparity)
        tf.summary.image('dispR', adas.dispR)
        tf.summary.image('dispLbar', adas.dispLbar)
        tf.summary.image('occmap', adas.occ)
        tf.summary.image('warpedLeft', adas.Ilbar)
        tf.summary.image('rawPicLeft', adas.img0)
        tf.summary.image('rawPicRight', adas.img1)
        
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./adasgraphs', sess.graph)
        
        saver = tf.train.Saver()
        
        epoch = 300
        episod = min(10**10,volumeImage)
        # load data and train adas
        steps = -1
        for iEpoch in range(epoch):
            for idx in range(episod):
                ''' train network in left right consist '''
                imgLlong, imgRlong = loadData.loadData(idx, imageList)
                imgR = imgRlong[0,:,1550-1472:,:]
                imgL = imgLlong[0,:,1550-1472:,:]
                imgR1 = np.fliplr(imgL)
                imgL1 = np.fliplr(imgR)
                imgR1 = np.expand_dims(imgR1, axis = 0)
                imgL1 = np.expand_dims(imgL1, axis = 0)
                dispR = sess.run(adas.disparity, {adas.img0: imgL1, adas.img1: imgR1, adas.img1long: imgRlong, adas.dispR: imgL1[:,:,:,0:1]})
                dispR = dispR[0,:,:,:]
                dispR = np.fliplr(dispR)
                dispR = np.expand_dims(dispR, axis=0)
                
                imgR2 = imgRlong[:,:,1550-1472:,:]
                imgL2 = imgLlong[:,:,1550-1472:,:]
                sess.run(adas.train, {adas.img0: imgL2, adas.img1: imgR2, adas.img1long: imgRlong, adas.dispR: dispR})
                
                steps += 1
                if steps % 100 == 0:
                    ttt = sess.run(merged, {adas.img0: imgL2, adas.img1: imgR2, adas.img1long: imgRlong, adas.dispR: dispR})
                    writer.add_summary(ttt, steps)
                    
                    saver.save(sess, './models/adasNet', global_step=steps)

    return 0


if __name__=='__main__':
    trainAdas()














           