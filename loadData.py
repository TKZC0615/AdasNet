#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:45:07 2017

@author: sensetime
"""

import os
from scipy import misc
import numpy as np
import csv

height_orig = 500
width_orig = 1700
height_input = 448
width_input = 1550#1472

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

'''===========================================
get Image list
    function:
        in mainDir
        in left image dir and right image dir
        find image pairs
    return:
        [ [leftImageName1 , leftImageName2 ,  ... ... , leftImageNameX ] ,
          [rightImageName1, rightImageName2,  ... ... , rightImageNameX] ]
==========================================='''
def getImagesList(dataset='sensetimeData'):#'cityscape'
    if dataset=='cityscape':
        mainDirs = ['/data/dataset/cityscape']
        leftDir = '/leftImg8bit'
        rightDir = '/rightImg8bit'
    elif dataset=='sensetimeData':
        mainDirs = ['/data/dataset/dw/dw_20170906_160228_0.000000_0.000000',
                   '/data/dataset/dw/dw_20170906_173602_0.000000_0.000000',
                   '/data/dataset/dw/dw_20170907_160012_0.000000_0.000000']
        leftDir = '/rectified_image_1'
        rightDir = '/rectified_image_2'
    elif dataset=='kitti':
        mainDirs = ['/data/dataset/stereo2015/data_scene_flow/testing',
                    '/data/dataset/stereo2015/data_scene_flow/training']
        leftDir = '/image_2'
        rightDir = '/image_3'
    elif dataset=='kitti2012':
        mainDirs = ['/data/dataset/stereo2012/data_stereo_flow/training']
        leftDir = '/colored_0'
        rightDir = '/colored_1'
    leftImges = []
    rightImages = []
    for mainDir in mainDirs:
        ''' search in the left part'''
        for dirStr, _, fileNames in os.walk(mainDir + leftDir):
            for filename in fileNames:
                leftImges.append(dirStr+'/'+filename)
        ''' search in the right part'''
        for dirStr, _, fileNames in os.walk(mainDir + rightDir):
            for filename in fileNames:
                rightImages.append(dirStr+'/'+filename)
    leftImges.sort()
    rightImages.sort()
    if len(leftImges)>=1500:
        leftImges = leftImges[:1500]
        rightImages = rightImages[:1500]
    ''' check pairs '''
    for i in range(len(leftDir)):
        lname = leftImges[i]
        rname = rightImages[i]
        tempStr = ''
        if dataset=='cityscape':
            tempStr = lname.replace('left', 'right')
        if dataset=='sensetimeData':
            tempStr = lname.replace('l', 'r')
            tempStr = tempStr.replace('image_1', 'image_2')
        if dataset=='kitti':
            tempStr = lname.replace('image_2', 'image_3')
        if dataset=='kitti2012':
            tempStr = lname.replace('colored_0', 'colored_1')
#        print(tempStr, rname)
        if rname != tempStr:
            return 0

    return [leftImges, rightImages]



'''===========================================
get Image data
    input:
        idx: img ID
        channel: 1 or 3 (gray or RGB)
    return:
        data Left : [1, height_input, width_input, channel]
        data Right: [1, height_input, width_input, channel]
==========================================='''


def loadData(idx, imageList, showType='normal', channel = 3):
    ''' read img L&R raw data '''
    imgL = misc.imread(imageList[0][idx])
    imgR = misc.imread(imageList[1][idx])
    ''' transformed as gray image '''
    if channel == 1:
        imgL = rgb2gray(imgL)
        imgR = rgb2gray(imgR)
    ''' resize image '''
    imgL = misc.imresize(imgL, (height_orig, width_orig, channel), interp='bilinear')
    imgR = misc.imresize(imgR, (height_orig, width_orig, channel), interp='bilinear')
    
    if showType == 'lr':
        imgLlr = np.fliplr(imgL)
        imgRlr = np.fliplr(imgR)
        imgL = imgRlr
        imgR = imgLlr
    elif showType == 'ud':
        imgLud = np.flipud(imgL)
        imgRud = np.flipud(imgR)
        imgL = imgLud
        imgR = imgRud
    
    ''' with random range and expand dim '''
    imgL = np.expand_dims(imgL, axis=0)
    imgR = np.expand_dims(imgR, axis=0)
    if channel == 1:
        imgL = np.expand_dims(imgL, axis=3)
        imgR = np.expand_dims(imgR, axis=3)
    randomx = np.random.randint(0, width_orig - width_input - 1 )
    randomy = np.random.randint(0, height_orig - height_input - 1)
    imgL = imgL[:, randomy : randomy + height_input, randomx : randomx + width_input, :]
    imgR = imgR[:, randomy : randomy + height_input, randomx : randomx + width_input, :]
    
    return imgL, imgR#[imgL,imgRlr,imgLud], [imgR,imgLlr,imgRud]

if __name__=='__main__':
    imagesList = getImagesList()

































