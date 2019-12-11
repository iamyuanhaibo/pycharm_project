#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/12/2 17:30
# !@Author:yuan
# !@File:new_trackbarToGetParam.py

import cv2
import numpy as np

'''
#图片的自动白平衡算法,灰度世界算法
@输入参数:
	img:彩色图片
@返回值:
	white_balanced_img:白平衡后的图片,彩色图片
'''
def white_balance(img):
	#拆分b,g,r通道
	b, g, r = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
	#b,g,r3个通道的平均值k
	b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
	k = (b_avg + g_avg + r_avg) / 3
	#各个通道的增益
	kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
	#每个通道的每个像素新值
	new_b = (b * kb)
	new_g = (g * kg)
	new_r = (r * kr)

	#溢出(>255)处理,设为255
	for i in range(len(new_b)):
		for j in range(len(new_b[0])):
			new_b[i][j] = 255 if new_b[i][j] > 255 else new_b[i][j]
			new_g[i][j] = 255 if new_g[i][j] > 255 else new_g[i][j]
			new_r[i][j] = 255 if new_r[i][j] > 255 else new_r[i][j]

	# print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
	white_balanced_img = np.uint8(np.zeros_like(img))
	white_balanced_img[:, :, 0] = new_b
	white_balanced_img[:, :, 1] = new_g
	white_balanced_img[:, :, 2] = new_r
	return white_balanced_img

'''
#去除图片的高光区域
@输入参数:
	img:彩色图片
	highlight_gray:高光区域灰度值的设置,可调整,默认=220
@返回值:
	inpaired_timg:已去除高光区域的彩色图片
'''
def remove_highlight(img,highlight_gray = 220):
	#转成灰度图片
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#寻找高光区域
	_, biny = cv2.threshold(gray_img, highlight_gray, 255, cv2.THRESH_BINARY)
	#对找到的高光区域进行膨胀
	dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	biny_dilate = cv2.dilate(biny,dilate_kernel,iterations=1)#th1,dilate_kernel,iterations=1,3
	#对高光区域进行修复,使用的是Alexandru Telea方法
	inpaired_timg = cv2.inpaint(img,biny_dilate,9,cv2.INPAINT_TELEA)
	return inpaired_timg

def nothing(x):
    pass

if __name__ =='__main__':

    #D:\PyCharm\previous_video_frames\out7frames
    test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1140.png'
    img1 = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
    #白平衡
    img2 = white_balance(img1)

    img = remove_highlight(img2,highlight_gray = 220)

    # 中值滤波 去除椒盐噪声
    # medianblur = cv2.medianBlur(img,5)#0:旧
    medianblur = cv2.GaussianBlur(img, (7, 7), 0)

    HsvImage = cv2.cvtColor(medianblur, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('GetHSVParam',cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('Hmin', 'GetHSVParam', 0, 180, nothing)
    cv2.createTrackbar('Hmax', 'GetHSVParam', 0, 180, nothing)

    cv2.createTrackbar('Smin', 'GetHSVParam', 0, 255, nothing)
    cv2.createTrackbar('Smax', 'GetHSVParam', 0, 255, nothing)

    cv2.createTrackbar('Vmin', 'GetHSVParam', 0, 255, nothing)
    cv2.createTrackbar('Vmax', 'GetHSVParam', 0, 255, nothing)


    cv2.namedWindow('another_hsv_param', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Hmin1', 'another_hsv_param', 0, 180, nothing)
    cv2.createTrackbar('Hmax1', 'another_hsv_param', 0, 180, nothing)

    cv2.createTrackbar('Smin1', 'another_hsv_param', 0, 255, nothing)
    cv2.createTrackbar('Smax1', 'another_hsv_param', 0, 255, nothing)

    cv2.createTrackbar('Vmin1', 'another_hsv_param', 0, 255, nothing)
    cv2.createTrackbar('Vmax1', 'another_hsv_param', 0, 255, nothing)

    cv2.namedWindow('mask0_or_mask1', cv2.WINDOW_AUTOSIZE)

    while(1):
        cv2.imshow('Source Img',img)

        #红色HSV阈值1
        hmin = cv2.getTrackbarPos('Hmin', 'GetHSVParam')
        hmax = cv2.getTrackbarPos('Hmax', 'GetHSVParam')
        smin = cv2.getTrackbarPos('Smin', 'GetHSVParam')
        smax = cv2.getTrackbarPos('Smax', 'GetHSVParam')
        vmin = cv2.getTrackbarPos('Vmin', 'GetHSVParam')
        vmax = cv2.getTrackbarPos('Vmax', 'GetHSVParam')

        #红色HSV阈值2
        hmin1 = cv2.getTrackbarPos('Hmin1', 'another_hsv_param')
        hmax1 = cv2.getTrackbarPos('Hmax1', 'another_hsv_param')
        smin1 = cv2.getTrackbarPos('Smin1', 'another_hsv_param')
        smax1 = cv2.getTrackbarPos('Smax1', 'another_hsv_param')
        vmin1 = cv2.getTrackbarPos('Vmin1', 'another_hsv_param')
        vmax1 = cv2.getTrackbarPos('Vmax1', 'another_hsv_param')

        '''
        #mask0
        '''
        rhsv_lower0 = np.array([hmin,smin,vmin],dtype=np.uint8)
        rhsv_upper0 = np.array([hmax,smax,vmax],dtype=np.uint8)
        #根据阈值构建掩模
        mask0 = cv2.inRange(HsvImage,rhsv_lower0,rhsv_upper0)
        # cv2.imshow('HSV',mask)

        #1
        hsv_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_ELLIPSE, (10, 10)
        mask0_open = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, hsv_open_kernel)

        #1
        hsv_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_RECT,(10,10)
        mask0_open_dilate = cv2.dilate(mask0_open, hsv_dilate_kernel, iterations=1)  # iterations=1

        #第二种，彩色掩膜方式,在原图上显示
        mask0_or_img = cv2.bitwise_or(img,img,mask = mask0_open_dilate)
        cv2.imshow('GetHSVParam',mask0_or_img)

        '''
        #mask1
        '''
        rhsv_lower1 = np.array([hmin1,smin1,vmin1],dtype=np.uint8)
        rhsv_upper1 = np.array([hmax1,smax1,vmax1],dtype=np.uint8)
        #根据阈值构建掩模
        mask1 = cv2.inRange(HsvImage,rhsv_lower1,rhsv_upper1)
        # cv2.imshow('HSV',mask)

        mask1_open = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, hsv_open_kernel)
        mask1_open_dilate = cv2.dilate(mask1_open, hsv_dilate_kernel, iterations=1)

        mask1_or_img = cv2.bitwise_or(img,img,mask = mask1_open_dilate)
        cv2.imshow('another_hsv_param',mask1_or_img)

        #final mask
        # mask = mask0+mask1
        mask = cv2.bitwise_or(mask0, mask1)

        mask_or_img = cv2.bitwise_or(img,img,mask = mask)
        cv2.imshow('mask0_or_mask1',mask_or_img)



        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


