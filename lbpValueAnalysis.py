#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/23 20:18
# !@Author:yuan
# !@File:lbpValueAnalysis.py


import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

img = cv2.imread('D:\\PyCharm\\pycharm_project\\6000.png',cv2.IMREAD_COLOR)
cv2.imshow('Source Img',img)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Img',grayImage)

imageClone = img.copy()
imageClone1 = img.copy()
clone = grayImage.copy()


'''
#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
'''
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])


#img:RGB image
# radius = 1
# n_points = 8*radius
def uniformLBP(img,radius = 3,n_points = 24):
    # print(img.shape[2]) #图像的通道数
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(grayImage.dtype)
    # cv2.imshow('ROI1GrayImage', grayImage)
    lbp = local_binary_pattern(grayImage, n_points, radius, method='uniform')
    # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
    lbpInt = lbp.astype(np.uint8)
    # cv2.imshow('ROILBPImage', lbp)
    # print(lbp[0:3])
    # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
    # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
    opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
    # print(opencv_hist.dtype)
    opencv_hist /= opencv_hist.sum()
    # print(opencv_hist)
    # plt.plot(opencv_hist, 'g')
    # plt.show()
    return opencv_hist

# uniformLBP(img)

#确认感兴趣区域的坐标，画出来就能知道坐标了
#背景区域
# cv2.rectangle(img,(0,0),(200,90),(0,0,255),1) #(x,y)
#积水区域
cv2.rectangle(img,(150,160),(350,250),(0,0,255),1)
cv2.imshow('ROI area on the image',img)


#背景区域
# ROI = clone[0:90,0:200] #[y1:y2,x1:x2] 水：clone[160:250,150:350]
#积水区域
ROI = clone[160:250,150:350]
cv2.imshow('Only ROI area',ROI)


#16*16的目标图片
realROI = imageClone[160:176,160:176]
cv2.imshow('Real ROI area',realROI)
realROILBPHist = uniformLBP(realROI,radius = 3,n_points = 24)
# print(realROILBPHist)
# plt.plot(realROILBPHist, 'g')
# plt.show()

#16*16的待查找图片示例
# realROI1 = imageClone[176:192,166:182]
# cv2.imshow('Real ROI area1',realROI1)
# realROILBPHist1 = uniformLBP(realROI1)
# print(realROILBPHist1)
# plt.plot(realROILBPHist1, 'g')
# plt.show()

# similaTwo = (realROILBPHist1-realROILBPHist)**2
# print(similaTwo)


leftUpXy = []

'''
#滑动窗口，计算每个滑动窗口的纹理特征值
'''
(winW,winH) = (16,16) #(5,5)
stepSize = 1 #3
for (x,y,window) in slidingWindow(imageClone,stepSize = stepSize,windowSize=(winW,winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    #窗口切片
    slice = imageClone[y:y+winH,x:x+winW]
    # cv2.namedWindow('slidingSlice',0)
    # cv2.imshow('slidingSlice',slice)

    tempUniformLBPHist = uniformLBP(slice,radius = 3,n_points = 24)

    corr = cv2.compareHist(realROILBPHist, tempUniformLBPHist, cv2.HISTCMP_CORREL)

    #欧式距离度量
    # cha = tempUniformLBPHist-realROILBPHist
    #(cha*cha).sum()
    if corr > 0.6:
        xy = (x,y)
        leftUpXy.append(xy)
        imageClone1[y:y+winH,x:x+winW] = [0,0,0]



# print(leftUpXy)
print(len(leftUpXy))
cv2.imshow('Changed Image',imageClone1)




if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()
