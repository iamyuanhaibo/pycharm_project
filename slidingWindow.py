#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/4 15:42
# !@Author:yuan
# !@File:slidingWidow.py

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops


'''
计算灰度共生矩阵四个方向（0,45,90,135）的均值灰度共生矩阵
注意:inds需要为灰度图像，两像素距离默认为1
'''
# def getGrayCooMatAverage(grayImage):
#     global maxValue
#     grayCooMat0 = greycomatrix(grayImage, [1], [0], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat45 = greycomatrix(grayImage, [1], [np.pi / 4], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat90 = greycomatrix(grayImage, [1], [np.pi / 2], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat135 = greycomatrix(grayImage, [1], [3 * np.pi / 4], levels=maxValue, normed=True, symmetric=False)
#
#     grayCooMatAverage = (grayCooMat0 + grayCooMat45 + grayCooMat90 + grayCooMat135) / 4
#     return grayCooMatAverage


#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])

img = cv2.imread('D:\\PyCharm\\pycharm_project\\lena.jpg',cv2.IMREAD_COLOR)
cv2.imshow('Source Img',img)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Img',grayImage)

grayCopy = grayImage.copy()

# bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
# inds = np.digitize(clone,bins)

# print(inds[0:3])
# maxValue = inds.max()+1
# print(maxValue)

#类型转换，inds为int64,将其转为uint8，使符合cv2.imshow的数据类型要求，使之可视化
# indsUint8 = inds.astype(np.uint8)
# cv2.imshow('Inds0',indsUint8)
# print('最大值为：',indsUint8.max())

# energyList = []
# leftUpXYList = []

(winW,winH) = (80,80) #(80,80)
stepSize = 16
for (x,y,window) in slidingWindow(grayCopy,stepSize = stepSize,windowSize=(winW,winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    # print(window.sum())
    clone = grayCopy.copy()
    #画小窗口，可视化
    cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,255,0),2)
    cv2.imshow('Window',clone)
    # cv2.waitKey(50)
    #窗口切片显示
    slice = grayCopy[y:y+winH,x:x+winW]
    cv2.imshow('slidingSlice',slice)
    cv2.waitKey(50)

    # grayCooMatAverage = getGrayCooMatAverage(slice)
    # energy = greycoprops(grayCooMatAverage, 'energy')
    # leftUpXY = (y,x)
    # energyList.append(energy)
    # if energy >= 0.3 and energy <= 0.4:
    #     leftUpXYList.append(leftUpXY)

    # energy = greycoprops(grayCooMatAverage, 'energy')
    # if energy > 0.3:
    #     indsUint8[y:y+winH,x:x+winW] = [0]
    #     # cv2.imshow('inds ROI2', indsUint8ROI)
    #     # cv2.imshow('inds', indsUint8)

# print(energyList)
# print(len(energyList))
#
# print(leftUpXYList)
# print(len(leftUpXYList))

#对energy >= 0.3 and energy <= 0.4的块儿,置成白色
# for i in range(len(leftUpXYList)):
#     indsUint8[leftUpXYList[i][0]:leftUpXYList[i][0]+winH,leftUpXYList[i][1]:leftUpXYList[i][1]+winW] = 255
# cv2.imshow('Inds1', indsUint8)

# energy = greycoprops(grayCooMatAverage, 'energy')
# contrast = greycoprops(grayCooMatAverage,'contrast')
# correlation = greycoprops(grayCooMatAverage,'correlation')
# homogeneity = greycoprops(grayCooMatAverage,'homogeneity')

# print(energy)

if cv2.waitKey(1) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()