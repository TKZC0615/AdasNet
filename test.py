#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:32:51 2017

@author: sensetime
"""

import tensorflow as tf
import numpy as np
from scipy import misc
from AdasNet import AdasNet
from loadData import getImagesList
import cv2
import os

mainDir = '/data/showAdas'
estLeft = mainDir + '/city_warpedleft'
gtLeft = '/data/dataset/cityscape/leftImg8bit'

height_input = 448
width_input = 1472
width_orig = 1550
channel = 3

leftDir = './input/left'
rightDir = './input/right'

def loadTestData(filename):
    ''' read img L&R raw data '''
    imgL = misc.imread(leftDir+'/'+filename)
    imgR = misc.imread(rightDir+'/'+filename)
    ''' resize image '''
    imgL = misc.imresize(imgL, (height_input, width_input, channel), interp='bilinear')
    imgR = misc.imresize(imgR, (height_input, width_input, channel), interp='bilinear')
    ''' with random range and expand dim '''
    imgL = np.expand_dims(imgL, axis=0)
    imgR = np.expand_dims(imgR, axis=0)
    
    return imgL, imgR

def showImage():
    sess = tf.Session()
    
    adas = AdasNet()
    
    new_saver = tf.train.Saver()
    new_saver.restore(sess, './models/adasNet-122500')
    
    for filename in os.listdir(leftDir):
            
        imgL, imgR = loadTestData(filename)
        imgRlong = misc.imresize(imgR[0], (height_input, width_orig, channel), interp='bilinear')
        imgRlong = np.expand_dims(imgRlong, axis = 0)
        # get Ilbar
        Ilbar = sess.run(adas.Ilbar, {adas.img0: imgL, adas.img1: imgR, adas.img1long: imgRlong, adas.dispR: imgL[:,:,:,0:1]})
        Ilbar = Ilbar[0,:,:,:]
        Ilbar = Ilbar*1.0
        # get occ
        occ = sess.run(adas.occ, {adas.img0: imgL, adas.img1: imgR, adas.img1long: imgRlong, adas.dispR: imgL[:,:,:,0:1]})
        occ = occ[0,:,:,0]
        # get disparity
        disp = sess.run(adas.disparity, {adas.img0: imgL, adas.img1: imgR, adas.img1long: imgRlong, adas.dispR: imgL[:,:,:,0:1]})
        disp = disp[0,:,:,0]
        
        ''' saving'''
        fileName = filename
        misc.imsave('./output' + '/warpedleft/' + fileName, Ilbar)
        misc.imsave('./output' + '/occ/' + fileName, occ)
        misc.imsave('./output' + '/disp/' + fileName, disp)
        print('raw :'+ filename +'done!')

if __name__=='__main__':
    showImage()



