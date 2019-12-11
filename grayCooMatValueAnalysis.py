#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/11 9:49
# !@Author:yuan
# !@File:textureFeatureValueAnalysis.py

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import matplotlib.pyplot as plt


'''
计算灰度共生矩阵四个方向（0,45,90,135）的均值灰度共生矩阵
注意:inds需要为灰度图像，两像素距离默认为1
'''
def getGrayCooMatAverage(grayImage):
    global maxValue
    grayCooMat0 = greycomatrix(grayImage, [1], [0], levels=maxValue, normed=True, symmetric=False)
    grayCooMat45 = greycomatrix(grayImage, [1], [np.pi / 4], levels=maxValue, normed=True, symmetric=False)
    grayCooMat90 = greycomatrix(grayImage, [1], [np.pi / 2], levels=maxValue, normed=True, symmetric=False)
    grayCooMat135 = greycomatrix(grayImage, [1], [3 * np.pi / 4], levels=maxValue, normed=True, symmetric=False)

    grayCooMatAverage = (grayCooMat0 + grayCooMat45 + grayCooMat90 + grayCooMat135) / 4
    return grayCooMatAverage

'''
#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
'''
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])
#'D:\\PyCharm\\pycharm_project\\lena.jpg'

#D:\PyCharm\pycharm_project\textureVideoSource\afternoonWithoutWaterFrames

img = cv2.imread('D:\\PyCharm\\pycharm_project\\textureVideoSource\\afternoonWithoutWaterFrames\\600.png',cv2.IMREAD_COLOR)
cv2.imshow('Source Img',img)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Img',grayImage)

clone = grayImage.copy()
bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
#将灰度级降为0-16
inds = np.digitize(clone,bins)

#类型转换，inds为int64,将其转为uint8，使符合cv2.imshow的数据类型要求，使之可视化
indsUint8 = inds.astype(np.uint8)
cv2.imshow('Inds0',indsUint8)

energyList = []
#用于灰度共生矩阵的输入参数，灰度最大级数
maxValue = inds.max()+1
'''
#滑动窗口，计算每个滑动窗口的纹理特征值
'''
(winW,winH) = (5,5) #(5,5)
stepSize = 5 #3

for (x,y,window) in slidingWindow(indsUint8,stepSize = stepSize,windowSize=(winW,winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    #窗口切片
    slice = indsUint8[y:y+winH,x:x+winW]
    cv2.namedWindow('slidingSlice',0)
    cv2.imshow('slidingSlice',slice)
    grayCooMatAverage = getGrayCooMatAverage(slice)
    energy = greycoprops(grayCooMatAverage, 'energy')
    energyList.append(energy)

#去除多重方括号
energyList = np.array(energyList).flatten()
energyList = list(energyList)

print(energyList)
print(len(energyList))

plt.plot(energyList,'k--')
plt.show()


if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()
