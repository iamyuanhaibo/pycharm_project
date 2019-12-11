#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/27 11:42
# !@Author:yuan
# !@File:trackbarToGetParam.py

import cv2
import numpy as np

def nothing(x):
    pass

if __name__ =='__main__':
    #0:最原始的版本
    #D:\PyCharm\previous_video_frames\out7frames
    test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1280.png'
    img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)

    # 中值滤波 去除椒盐噪声
    # medianblur = cv2.medianBlur(img,5)#0:旧
    medianblur = cv2.GaussianBlur(img, (7, 7), 0)

    HsvImage = cv2.cvtColor(medianblur, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('GetHSVParam')

    cv2.createTrackbar('Hmin', 'GetHSVParam', 0, 180, nothing)
    cv2.createTrackbar('Hmax', 'GetHSVParam', 0, 180, nothing)

    cv2.createTrackbar('Smin', 'GetHSVParam', 0, 255, nothing)
    cv2.createTrackbar('Smax', 'GetHSVParam', 0, 255, nothing)

    cv2.createTrackbar('Vmin', 'GetHSVParam', 0, 255, nothing)
    cv2.createTrackbar('Vmax', 'GetHSVParam', 0, 255, nothing)

    while(1):
        cv2.imshow('Source Img',img)

        hmin = cv2.getTrackbarPos('Hmin', 'GetHSVParam')
        hmax = cv2.getTrackbarPos('Hmax', 'GetHSVParam')
        smin = cv2.getTrackbarPos('Smin', 'GetHSVParam')
        smax = cv2.getTrackbarPos('Smax', 'GetHSVParam')
        vmin = cv2.getTrackbarPos('Vmin', 'GetHSVParam')
        vmax = cv2.getTrackbarPos('Vmax', 'GetHSVParam')

        rhsv_lower = np.array([hmin,smin,vmin],dtype=np.uint8)
        rhsv_upper = np.array([hmax,smax,vmax],dtype=np.uint8)
        #根据阈值构建掩模
        mask = cv2.inRange(HsvImage,rhsv_lower,rhsv_upper)
        # cv2.imshow('HSV',mask)

        #0
        # #开运算,去除孤立噪声
        # #卷积核(5,5)表示长宽均不足5的孤立噪声点将被去除
        Kernel = np.ones((6,6),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,Kernel)

        #1
        # hsv_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_ELLIPSE, (10, 10)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hsv_open_kernel)
        #0
        # # cv2.imshow('After Opening',mask)
        # #膨胀,填充孔
        Kernel1 = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(mask,Kernel1,iterations = 1)

        #1
        # hsv_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_RECT,(10,10)
        # dilation = cv2.dilate(mask, hsv_dilate_kernel, iterations=1)  # iterations=1

        #第一种黑白掩膜方式
        # cv2.imshow('GetHSVParam',dilation)

        #第二种，彩色掩膜方式
        mask = dilation
        OrImage = cv2.bitwise_or(img,img,mask = mask)
        cv2.imshow('GetHSVParam',OrImage)

        # contours,hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #dilation
        # contours_image = cv2.drawContours(img,contours,-1,(255,255,255),1) #img,contours,-1,(255,255,255),1
        # cv2.imshow('HSV',contours_image)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()















