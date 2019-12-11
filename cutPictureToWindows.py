#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/31 15:52
# !@Author:yuan
# !@File:cutPictureToWindows.py

'''
#将样本图片以16*16的窗口滑动，步长为16并依次保存
'''
import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

'''
#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
'''
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])

'''
#使用滑动窗口将样本以步长为16截成16*16的小灰度图片
@输入参数:
    sample_path:样本路径,例:sample_path = 'D:\\samples_withcolor\\neg\\'
    sample_num:样本数
    save_file_path:保存的16*16的小灰度图片的路径,例:'D:\\samples_withcolor\\neg_sixteen_gray_step16\\'
    winW:滑动窗口宽度
    winH:滑动窗口高度
    stepsize:滑动窗口步长
@返回值:
    无,打印操作完成提示信息
'''
def save_image_to_files(sample_path,sample_num,save_file_path,winW=16,winH=16,stepSize=16):
    i = 1
    for j in range(1,(sample_num+1)):
        img = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_COLOR)
        cv2.imshow('Source Img',img)
        '''
        #滑动窗口，依次保存窗口图片
        '''
        for (x,y,window) in slidingWindow(img,stepSize = stepSize,windowSize=(winW,winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            #窗口切片
            slice = img[y:y+winH,x:x+winW]
            # gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_file_path + str(i) + '.png',slice)
            cv2.imshow('slidingSlice',slice)
            cv2.waitKey(10)
            i = i + 1
    print('Save 16 times 16 image to files,step 8,done.')

if __name__ =='__main__':
    '''
    #负样本
    '''
    # #样本数
    # neg_sample_num = 253
    # #原负样本路径:D:\samples_withcolor\neg
    # neg_sample_path = 'D:\\samples_withcolor\\neg\\'
    # #保存的负样本路径:D:\samples_withcolor\neg_sixteen_gray_step16
    # neg_save_file_path = 'D:\\samples_withcolor\\neg_sixteen_gray_step16\\'
    # save_image_to_files(neg_sample_path,neg_sample_num,neg_save_file_path,winW=16,winH=16,stepSize=16)

    '''
    #正样本
    '''
    # #样本数
    # pos_sample_num = 102
    # #原负样本路径:D:\samples_withcolor\pos
    # pos_sample_path = 'D:\\samples_withcolor\\pos\\'
    # #保存的负样本路径:D:\samples_withcolor\pos_sixteen_gray_step16
    # pos_save_file_path = 'D:\\samples_withcolor\\pos_sixteen_gray_step6\\'

    #D:\samples_withcolor\sample_optim\pos
    sample_path = 'D:\\samples_withcolor\\sample_optim\\pos\\'
    sample_num = 195
    #D:\samples_withcolor\sample_optim\pos_sixteen_step8
    sample_save_path = 'D:\\samples_withcolor\\sample_optim\\pos_sixteen_step8\\'

    save_image_to_files(sample_path,sample_num,sample_save_path,winW=16,winH=16,stepSize=8)

    if cv2.waitKey(0) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
