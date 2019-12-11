#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/9/16 16:08
# !@Author:yuan
# !@File:textToGetImageArea.py

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix,greycoprops


# D:\\PyCharm\\pycharm_project\\5.jpg
img = cv2.imread('D:\\PyCharm\\pycharm_project\\lena.jpg',cv2.IMREAD_COLOR)

#画线
# cv2.line(img,(0,165),(440,165),(0,0,255),1)
# cv2.line(img,(220,0),(220,330),(0,0,255),1)
cv2.imshow('Source Img',img)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Img',grayImage)
# print(grayImage[0:3])

#图像分块操作
# #ROI 区域
# img1 = img[0:165,0:220]
# cv2.imshow('ROI1',img1)
# img2 = img[0:165,221:440]
# cv2.imshow('ROI2',img2)
# img3 = img[166:330,0:220]
# cv2.imshow('ROI3',img3)
# img4 = img[166:330,221:440]
# cv2.imshow('ROI4',img4)

# radius = 1
# n_points = 8*radius
def uniformLBP(img,radius = 3,n_points = 24):
    # print(img.shape[2]) #图像的通道数
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(grayImage.dtype)
    cv2.imshow('ROI1GrayImage', grayImage)
    lbp = local_binary_pattern(grayImage, n_points, radius, method='uniform')
    # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
    lbpInt = lbp.astype(np.uint8)
    cv2.imshow('ROILBPImage', lbp)
    print(lbp[0:3])
    # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
    # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
    opencv_hist = cv2.calcHist([lbpInt], [0], None, [(n_points+2)], [0, (n_points+2)])
    # print(opencv_hist.dtype)
    opencv_hist /= opencv_hist.sum()
    # print(opencv_hist)
    plt.plot(opencv_hist, 'g')
    plt.show()

def originLBP(img,radius = 1,n_points = 8):
    # print(img.shape[2]) #图像的通道数
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(grayImage.dtype)
    cv2.imshow('ROI1GrayImage', grayImage)
    lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
    # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
    lbpInt = lbp.astype(np.uint8)
    cv2.imshow('ROILBPImage', lbp)
    print(lbp[0:3])
    # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
    # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
    opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
    # print(opencv_hist.dtype)
    opencv_hist /= opencv_hist.sum()
    print(opencv_hist[0:10])
    plt.plot(opencv_hist, 'g')
    plt.show()

# uniformLBP(img)
# originLBP(img)

'''
计算灰度共生矩阵四个方向（0,45,90,135）的均值灰度共生矩阵
注意:inds需要为灰度图像，两像素距离默认为1
'''
def getGrayCooMatAverage(inds):
    grayCooMat0 = greycomatrix(inds, [1], [0], levels=maxValue, normed=True, symmetric=False)
    grayCooMat45 = greycomatrix(inds, [1], [np.pi / 4], levels=maxValue, normed=True, symmetric=False)
    grayCooMat90 = greycomatrix(inds, [1], [np.pi / 2], levels=maxValue, normed=True, symmetric=False)
    grayCooMat135 = greycomatrix(inds, [1], [3 * np.pi / 4], levels=maxValue, normed=True, symmetric=False)

    grayCooMatAverage = (grayCooMat0 + grayCooMat45 + grayCooMat90 + grayCooMat135) / 4
    return grayCooMatAverage

#基于灰度共生矩阵计算出的特征：对比度，异样性，同质性，能量，角二阶矩，相关度
#灰度共生矩阵四个常用特征:对比度，相关度，能量，熵
def contrastFeature(matrixCoo):
    contrast = greycoprops(matrixCoo,'contrast')
    return contrast
#
def dissimilarityFeature(matrixCoo):
    dissimilarity = greycoprops(matrixCoo,'dissimilarity')
    return dissimilarity
#
def homogeneityFeature(matrixCoo):
    homogeneity = greycoprops(matrixCoo,'homogeneity')
    return homogeneity
#
def energyFeature(matrixCoo):
    energy = greycoprops(matrixCoo,'energy')
    return energy
#
def ASMFeature(matrixCoo):
    ASM = greycoprops(matrixCoo,'ASM')
    return ASM
#
def correlationFeature(matrixCoo):
    correlation = greycoprops(matrixCoo,'correlation')
    return correlation

#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])

#
bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
inds = np.digitize(grayImage,bins)

'''
#类型转换，inds为int64,将其转为uint8，使符合cv2.imshow的数据类型要求，使之可视化
'''
indsUint8 = inds.astype(np.uint8)
# print(inds.dtype)
# print(indsUint8.dtype)
cv2.imshow('Blured Img',indsUint8)


(winW,winH) = (80,80)
stepSize = 16
for (x,y,window) in slidingWindow(indsUint8,stepSize = stepSize,windowSize=(winW,winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    # print(window.sum())
    clone = indsUint8.copy()
    #窗口切片
    slice = clone[y:y+winH,x:x+winW]
    cv2.namedWindow('slidingSlice',0)
    cv2.imshow('slidingSlice',slice)
    # cv2.waitKey(50)



    #画小窗口，可视化
    cv2.rectangle(clone,(x,y),(x+winW,y+winH),(255,0,0),2)
    cv2.imshow('Window',clone)
    cv2.waitKey(50)









'''
#灰度级数：0-16（灰度∈[0,255]），所以为17级，要加一
'''
# maxValue = inds.max()+1

'''
求取灰度共生矩阵的输入参数
inds:输入图像
[1]:两个像素的距离
[np.pi/4]:角度
level:灰度级+1
normed:归一化
'''
# matrixCoo = greycomatrix(inds,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels = maxValue,normed=False,symmetric=False)
#
# grayCooMatAverage = getGrayCooMatAverage(inds)
#
# #灰度共生矩阵四个常用特征:能量，对比度，相关度，熵
# Energy = energyFeature(grayCooMatAverage)
# Contrast = contrastFeature(grayCooMatAverage)
# Correlation = correlationFeature(grayCooMatAverage)
# Homogeneity = homogeneityFeature(grayCooMatAverage)
#
# print('\nEnergy:',Energy,'\nContrast:',Contrast,'\nHomogeneity:',Homogeneity,'\nCorrelation:',Correlation)









if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()