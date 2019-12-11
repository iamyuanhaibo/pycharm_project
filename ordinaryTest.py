#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/23 21:36
# !@Author:yuan
# !@File:ordinaryTest.py

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

from new_trackbarToGetParam import white_balance


#设置矩阵在pycharm中全显示，不自动换行,防止写入txt数据格式不对
np.set_printoptions(linewidth=500000)
np.set_printoptions(threshold=np.inf)



'''
#验证两个矩阵相乘
'''
# L1 = np.array([[1,2,3],
#               [4,5,6]])
#
# L2 = np.array([[1,4,3],
#               [4,5,6]])
#
# print((L1-L2)**2)


'''
#验证列表添加坐标
'''
# l = []
#
# a = (1,2)
# l.append(a)
# b = (3,4)
# l.append(b)
#
# print(l)

'''
#验证np矩阵减法运算
'''
# a = np.array([[1,2,3,4]])
#
# b = np.array([[1,2,3,5]])
# print((a-b).sum())


'''
#LBP直方图相似度度量测试
'''
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\6000.png',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
#
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Img',grayImage)
#
# imageClone = img.copy()
# imageClone1 = img.copy()
# clone = grayImage.copy()
#
# def originLBP(img,radius = 1,n_points = 8):
#     # print(img.shape[2]) #图像的通道数
#     grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # print(grayImage.dtype)
#     # cv2.imshow('ROI1GrayImage', grayImage)
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     cv2.imshow('originLBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     # opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist[0:10])
#     plt.plot(opencv_hist, 'g')
#     plt.show()
#     return opencv_hist

#img:RGB image
# radius = 1
# n_points = 8*radius
# def uniformLBP(img,radius = 1,n_points = 8):
#     # print(img.shape[2]) #图像的通道数
#     grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # print(grayImage.dtype)
#     # cv2.imshow('ROI1GrayImage', grayImage)
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='uniform')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     cv2.imshow('uniformLBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     # opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist)
#     plt.plot(opencv_hist, 'g')
#     plt.show()
#     return opencv_hist

# a = originLBP(img,radius = 1,n_points = 8)
# b = uniformLBP(img,radius = 1,n_points = 8)



# #确认感兴趣区域的坐标，画出来就能知道坐标了
# #背景区域
# # cv2.rectangle(img,(0,0),(200,90),(0,0,255),1) #(x,y)
# #积水区域
# cv2.rectangle(img,(150,160),(350,250),(0,0,255),1)
#
# cv2.rectangle(img,(160,160),(176,176),(0,255,0),1)
# cv2.rectangle(img,(160,176),(176,192),(0,255,0),1)
#
# cv2.imshow('ROI area on the image',img)
#
# #背景区域
# # ROI = clone[0:90,0:200] #[y1:y2,x1:x2] 水：clone[160:250,150:350]
# #积水区域
# ROI = clone[160:250,150:350]
# cv2.imshow('Only ROI area',ROI)
#
# #16*16的目标图片
# realROI = imageClone[160:176,160:176]
# cv2.imshow('Real ROI area',realROI)
# realROILBPHist = uniformLBP(realROI,radius = 3,n_points = 24)
# # print(realROILBPHist)
# # print(realROILBPHist.sum())
# plt.plot(realROILBPHist, 'g')
# plt.show()
#
# # oriLBP = originLBP(realROI)
#
# # #16*16的待查找图片示例
# realROI1 = imageClone[176:192,160:176]
# cv2.imshow('Real ROI area1',realROI1)
# realROILBPHist1 = uniformLBP(realROI1,radius = 3,n_points = 24)
# # print(realROILBPHist1)
# plt.plot(realROILBPHist1, 'g')
# plt.show()
#
# '''
# #相似度度量指标
# '''
# #相关性:[0 1],越大越相似
# corr = cv2.compareHist(realROILBPHist,realROILBPHist1,cv2.HISTCMP_CORREL)
# print('Correlation:',corr)
#
# #卡方:[0 ∞],越小越相似
# chisqr = cv2.compareHist(realROILBPHist,realROILBPHist1,cv2.HISTCMP_CHISQR)
# print('Chisqr:',chisqr)
#
# #巴氏距离:[0 1],越小越相似
# bha = cv2.compareHist(realROILBPHist,realROILBPHist1,cv2.HISTCMP_BHATTACHARYYA)
# print('Bha:',bha)


'''
#LBP直方图的对比验证
'''
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\6000.png',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
#
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Img',grayImage)
#
# #均衡化操作,作为一个可选项,后面效果不好,可以试下
# # HistEqu = cv2.equalizeHist(grayImage)
# # cv2.imshow('Histogram Equalization Image',HistEqu)
#
# imageClone = img.copy()
# grayClone = grayImage.copy()
#
# def originLBP(grayImage,radius = 1,n_points = 8):
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     # cv2.imshow('originLBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     #归一化
#     opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist[0:10])
#     return opencv_hist
#
# #确认感兴趣区域的坐标，画出来就能知道大致坐标了
# #背景区域
# cv2.rectangle(img,(0,0),(200,90),(255,0,0),1) #(x,y)
# #积水区域
# cv2.rectangle(img,(150,160),(350,250),(0,0,255),1)
#
# #目标点坐标# objPy = 160 objPx = 160
# objPy = 160
# objPx = 160
# #待检测点坐标 comPy = 176 comPx = 160
# comPy = 200
# comPx = 200
#
# #积水区域小块,16*16
# cv2.rectangle(img,(objPx,objPy),((objPx+16),(objPy+16)),(0,255,0),1)
# cv2.rectangle(img,(comPx,comPy),((comPx+16),(comPy+16)),(0,255,0),1)
# cv2.imshow('ROI area on the image',img)
#
# #16*16的目标图片
# realROI = grayImage[objPy:(objPy+16),objPx:(objPx+16)]
# cv2.imshow('Real ROI area',realROI)
# realROILBPHist = originLBP(realROI,radius = 1,n_points = 8)
# plt.plot(realROILBPHist, 'g')
# plt.show()
#
# #16*16的待查找图片示例
# compROI = grayImage[comPy:(comPy+16),comPx:(comPx+16)] #
# cv2.imshow('Compare ROI area1',compROI)
# compROILBPHist = originLBP(compROI,radius = 1,n_points = 8)
# # print(realROILBPHist1)
# plt.plot(compROILBPHist, 'g')
# plt.show()

'''
#16进制打印
'''
# a = 18
# print('%#x'%a)

'''
#将图片切成16*16的小块并保存测试
'''
# '''
# #滑动窗口
# #yield的妙用，相当于return，不过是有需要就return，减少内存占用
# '''
# def slidingWindow(image,stepSize,windowSize):
#     for y in range(0,image.shape[0],stepSize):
#         for x in range(0,image.shape[1],stepSize):
#             yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])
#
# (winW,winH) = (16,16) #(5,5)
# stepSize = 16 #3
# i = 1
# cv2.imshow('Source Img',img)
# '''
# #滑动窗口，依次保存窗口图片
# '''
# for (x,y,window) in slidingWindow(img,stepSize = stepSize,windowSize=(winW,winH)):
#     if window.shape[0] != winH or window.shape[1] != winW:
#         continue
#     #窗口切片
#     slice = img[y:y+winH,x:x+winW]
#     cv2.imwrite('D:\\samples\\pos_sixteen\\' + str(i) + '.png',slice)
#     cv2.imshow('slidingSlice',slice)
#     cv2.waitKey(10)
#     i = i + 1

#
#
# slice = img[20:20+16,20:20+16]
# cv2.imshow('slidingSlice',slice)
#
# gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
# cv2.imshow('slidingSliceGray',gray)

# for i in range(5):

# i = 2
# img = cv2.imread('D:\\samples\\neg\\1.png', cv2.IMREAD_COLOR)
# cv2.imshow('Source Img'+' '+str(i), img)
#
#
# a = np.zeros((2,3,4))
# print(a)

'''
#opencv自带的SVM
'''
# svm = cv2.ml.SVM_create()
# svm.train()

'''
#写入txt文件操作
'''
# filename = 'D:\\samples\\test.txt'
# a = []
# # b = [0.352, 0.456, 0.785]
# for i in range(3):
#     with open(filename,'a') as file_object:
#         file_object.write(str(i)+'\n')#file_object.write(str(i)+'\n')
#         # file_object.write(str(b)+'\n')
# file_object.close()

'''
#读取txt文件测试,txt文件以\n为换行
'''
# with open(filename) as file_object1:
#     contents = file_object1.read()
#     print(contents)
#     file_object1.close()

'''
#open(filename,'a'),a属性测试
'''
# a = np.zeros((8423))
# for i in range(8423):
#     a[i] = 0
#
# print(a)
#
# filename = 'D:\\samples\\test.txt'
# with open(filename,'a') as file_object:
#     file_object.write(str(a))#file_object.write(str(i)+'\n')
#     # file_object.write(str(b)+'\n')
#     file_object.close()

# '''
# #写入单种正/负样本的特征对应的标记至txt文件,是一个一维矩阵[1,1,1,1,1,...,1]或[0,0,0,0,0,...,0]
# @输入参数:
#     sample_label_num:标记样本数
#     sample_label_path:最后生成的标记样本写入txt文件的路径及文件名称
#     label:标记,正样本为1,负样本为0
# @返回值：
#     无,最后打印写入标记完成信息
# '''
# def save_sample_label(sample_label_num,sample_label_path,label):
#     sample_label = np.zeros((sample_label_num))
#     for i in range(sample_label_num):
#         sample_label[i] = label
#     with open(sample_label_path, 'a') as file_object:
#         file_object.write(str(sample_label))  # file_object.write(str(i)+'\n')
#         file_object.close()
#     print('Write sample labels,done!')

#写入单种正/负样本的特征对应的标记至txt文件
# save_sample_label(pos_sample_label_num,pos_sample_label_path,pos_sample_label)

'''
#写入正负“混合”样本标记测试
'''
# def save_all_sample_labels(pos_sample_num,neg_sample_num,all_sample_label_path,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = np.zeros((pos_sample_num + neg_sample_num))
#     if negpos == 1:
#         for i in range(pos_sample_num + neg_sample_num):
#             if i <= (neg_sample_num-1):
#                 sample_label[i] = neg_label
#             else:
#                 sample_label[i] = pos_label
#     else:
#         for i in range(pos_sample_num + neg_sample_num):
#             if i <= (pos_sample_num-1):
#                 sample_label[i] = pos_label
#             else:
#                 sample_label[i] = neg_label
#     with open(all_sample_label_path, 'a') as file_object:
#         file_object.write(str(sample_label))  # file_object.write(str(i)+'\n')
#         file_object.close()
#
#     print('Write sample labels,done!')
#
# pos_sample_label_num = 4
# neg_sample_label_num = 3
# all_sample_label_path = 'D:\\samples\\test_all_label.txt'
#
# save_all_sample_labels(pos_sample_label_num,neg_sample_label_num,all_sample_label_path,pos_label = 1,neg_label = 0,negpos = 0)



'''
#读取txt文件测试,txt文件以\n为换行
'''
# filename = 'D:\\samples\\test.txt'

# a = []
# with open(filename) as file_object1:
#     contents = file_object1.readline()
#     a.append(contents)
#     print(a)
#     file_object1.close()

'''
#读取txt文件,并将其转为可用数据,未成功。。
'''
# a = []
# f = open(filename)
# lines = f.readlines()
# print(len(lines))
#
# for line in lines:
#     list = line.strip('\n').strip('[').strip(']')#list = line.strip('\n')
#     print(list)
#     a.append(list)
# print(a)

'''
#读取txt文件,txt文件以\n为换行
'''
# filename = 'D:\\samples\\all_sample_label.txt'
# with open(filename) as file_object1:
#     train_data_label = file_object1.read()
#     print(train_data_label)
#     file_object1.close()

# a = np.zeros((3,4))
# # print(a)
# a1 = np.array([1,2,3,4])
# a2 = np.array([3,4,5,6])
#
# a3 = np.array(([1,1,1,1],[1,1,1,1],[1,1,1,1]))
# a = a3
# # a[1] = a1
# # a[2] = a2
# print(a)

'''
#提取到图片特征到变量测试
'''
# '''
# #提取归一化后的原始LBP特征
# @输入参数：
#     grayImage:灰度图片
#     radius:lbp半径,默认=1
#     n_points:lbp点个数,默认=8
# @返回值:
#     归一化后的lbp直方图
# '''
# def originLBP(grayImage,radius = 1,n_points = 8):
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     # cv2.imshow('ROILBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     #归一化LBP直方图
#     opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist[0:10])
#
#     return opencv_hist #return opencv_hist
#
# '''
# #提取lbp特征,sample_num个[0.2,0.32,...,]256个元素
# @输入参数:
#     sample_path:待提取灰度图片(16*16)的路径
#     sample_num:待提取灰度图片的总数
# @返回值：
#     无,最后打印完成信息
# '''
# def extractFeature_saveToTxt(sample_path,sample_num):
#     train_hist = np.zeros((sample_num,256))
#     for j in range(1,sample_num+1):
#         grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
#         opencv_hist = originLBP(grayImage,radius = 1,n_points = 8)
#         opencv_hist = opencv_hist.flatten()
#         #只保留小数点后四位
#         opencv_hist = np.around(opencv_hist,4)
#         # print(opencv_hist)
#         # print(len(opencv_hist))
#         train_hist[j-1] = opencv_hist
#     return np.float32(train_hist)
#
# all_sample_num = 2
# all_sample_path = 'D:\\samples\\all_sixteen_gray\\'
#
# train_hist = np.zeros((all_sample_num,256))
# train_hist = extractFeature_saveToTxt(all_sample_path,all_sample_num)
#
# print(train_hist)
# print(len(train_hist))
# #查看数据类型是数组类型还是矩阵类型
# print(type(train_hist))
# #查看数据类型是浮点型还是整型
# print(train_hist.dtype)


# def save_all_sample_labels(pos_sample_num,neg_sample_num,all_sample_label_path,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = np.zeros((pos_sample_num + neg_sample_num))
#     if negpos == 1:
#         for i in range(pos_sample_num + neg_sample_num):
#             if i <= (neg_sample_num-1):
#                 sample_label[i] = neg_label
#             else:
#                 sample_label[i] = pos_label
#     else:
#         for i in range(pos_sample_num + neg_sample_num):
#             if i <= (pos_sample_num-1):
#                 sample_label[i] = neg_label
#             else:
#                 sample_label[i] = pos_label
#     with open(all_sample_label_path, 'a') as file_object:
#         file_object.write(str(sample_label))  # file_object.write(str(i)+'\n')
#         file_object.close()
#
#     print('Write sample labels,done!')


# train_label = np.ones((1,5))
# print(train_label)


# data = np.repeat(np.arange(1,2),5) #data = np.repeat(np.arange(1,2),5)[:,np.newaxis]
# print(data)
# print(type(data))

# def generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = np.repeat(np.arange(1,2),5)[:,np.newaxis]

# dat = np.ones(5)[:,np.newaxis]
# print(dat)

# a = []
# for i in range(5):
#     a.append(1)
# a = np.array(a)
# print(a)
# print(type(a))

'''
#生成特征标记测试
'''
# def generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = []
#     if negpos == 1:
#         for i in range(neg_sample_num):
#             sample_label.append(neg_label)
#         for j in range(pos_sample_num):
#             sample_label.append(pos_label)
#     else:
#         for i in range(pos_sample_num):
#             sample_label.append(pos_label)
#         for j in range(neg_sample_num):
#             sample_label.append(neg_label)
#     sample_label = np.array(sample_label)
#     sample_label = sample_label[:,np.newaxis]
#     return sample_label
#
# sample_label = generate_sample_label(2,3,pos_label = 1,neg_label = 0,negpos = 1)
# print(sample_label)
# print(type(sample_label))
# print(sample_label.dtype)

'''
#测试返回值,直接当变量测试
'''
# def aa():
#     a = np.array([1,2,3])
#     return a
# b = aa()
# print(b)

'''
#正确率的测试,mask = result == label的妙用
'''
# label = np.array([[0.],[0.],[0.],[0.],[0.]])
# result = np.array([[0.],[0.],[0.],[0.],[1.]])
# print(label.size)
# mask = result == label
# correct_percent = np.count_nonzero(mask)*100/label.size
# print(mask)
# print(correct_percent)

'''
#产生单独正/负样本的标记测试,及range(0)测试,不会进
'''
# def generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = []
#     if negpos == 1:
#         for i in range(neg_sample_num):
#             sample_label.append(neg_label)
#         for j in range(pos_sample_num):
#             sample_label.append(pos_label)
#     else:
#         for i in range(pos_sample_num):
#             sample_label.append(pos_label)
#         for j in range(neg_sample_num):
#             sample_label.append(neg_label)
#     sample_label = np.array(sample_label)
#     sample_label = sample_label[:,np.newaxis]
#     return sample_label

# a = generate_sample_label(3,0,pos_label = 1,neg_label = 0,negpos = 1)
# print(a)
# for i in range(0):
#     print('aa')
# print('bb')

'''
#SVM模型16*16的小图片验证测试
'''
# def originLBP(grayImage,radius = 1,n_points = 8):
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     # cv2.imshow('ROILBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     #归一化LBP直方图
#     opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist[0:10])
#     return opencv_hist #return opencv_hist
#
# def extract_lbpFeature_from_oneImage(grayImage):
#     train_hist = []
#     opencv_hist = originLBP(grayImage, radius=1, n_points=8)
#     opencv_hist = opencv_hist.flatten()
#     # 只保留小数点后四位
#     opencv_hist = np.around(opencv_hist, 4)
#     train_hist.append(opencv_hist)
#     train_hist = np.array(train_hist)
#     return np.float32(train_hist)
#
# #加载训练好的SVM模型
# svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\originlbp_svm_data.dat'
# svm_model_load = cv2.ml.SVM_load(svm_model_filepath)
#
# test_image_path = 'D:\\samples\\neg_sixteen_gray\\1.png'
# test_img = cv2.imread(test_image_path,cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Source Image',test_img)
#
# test_image_lbpfeature = extract_lbpFeature_from_oneImage(test_img)
# # print(test_image_lbpfeature)
# #测试样本的分类结果
# test_result = svm_model_load.predict(test_image_lbpfeature)[1]
# print(test_result)
# print(type(test_result))
# print(test_result.dtype)
#
# if int(test_result) == 1:
#     print('test_result == 1')
# else:
#     print('test_result == 0')


'''
#uniform LBP feature 测试
'''
# def uniformLBP(grayImage,radius = 1,n_points = 8):
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='uniform')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     # cv2.imshow('uniformLBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [10], [0, 10])#cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
# 	#归一化
#     opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist)
#     return opencv_hist
#
# def extract_UniformLBPFeature_saveToVar(sample_path,sample_num):
#     train_hist = np.zeros((sample_num,10))
#     for j in range(1,sample_num+1):
#         grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
#         opencv_hist = uniformLBP(grayImage,radius = 1,n_points = 8)
#         opencv_hist = opencv_hist.flatten()
#         #只保留小数点后四位
#         # opencv_hist = np.around(opencv_hist,4)
#         # print(opencv_hist)
#         # print(len(opencv_hist))
#         train_hist[j-1] = opencv_hist
#     return np.float32(train_hist)
#
# neg_sample_num = 10
# neg_save_file_path = 'D:\\samples_withcolor\\neg_sixteen_gray_step16\\'
#
# hist = extract_UniformLBPFeature_saveToVar(neg_save_file_path,neg_sample_num)
# print(hist)


# '''
# #sample_label测试
# '''
# def generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1):
#     sample_label = []
#     if negpos == 1:
#         for i in range(neg_sample_num):
#             sample_label.append(neg_label)
#         for j in range(pos_sample_num):
#             sample_label.append(pos_label)
#     else:
#         for i in range(pos_sample_num):
#             sample_label.append(pos_label)
#         for j in range(neg_sample_num):
#             sample_label.append(neg_label)
#     sample_label = np.array(sample_label)
#     sample_label = sample_label[:,np.newaxis]
#     return sample_label
#
# neg_sample_num = 3
# pos_sample_num = 2
# a = generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1)
# print(a)

# i = 10
# print(str(i+5))

# def a(b = 7):
# #     return b
# #
# # c = a(b=8)
# # print(c)

'''
#原始LBP直方图分析
'''
# def originLBP(grayImage,radius = 1,n_points = 8):
#     lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
#     # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
#     lbpInt = lbp.astype(np.uint8)
#     # cv2.imshow('ROILBPImage', lbp)
#     # print(lbp[0:3])
#     # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
#     # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
#     opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
#     # print(opencv_hist.dtype)
#     #归一化LBP直方图
#     opencv_hist /= opencv_hist.sum()
#     # print(opencv_hist[0:10])
#     return opencv_hist #return opencv_hist
#
#
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\6000.png',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
#
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Img',grayImage)
#
# opencv_hist = originLBP(grayImage,radius = 1,n_points = 8)
# plt.plot(opencv_hist, 'g')
# plt.show()

'''
#print变量测试
'''
# a = 10
# correct = 20
# print('a:',a,'correct:',correct)
# print('a:',a,'correct:',correct)

'''
#俩图片相减后,出现负值,按比例缩放,但效果不行
'''
# print(sub2.shape)
# height = sub2.shape[0]
# weight = sub2.shape[1]
# channels = sub2.shape[2]
# sub_max = sub2.max()
# sub_min = sub2.min()
# print('max:',sub2.max())
# print('min:',sub2.min())
#255*sub2[row,col,cha]/(sub_max-sub_min)变换到0-255
# for row in range(height):
# 	for col in range(weight):
# 		for cha in range(channels):
# 			sub2[row,col,cha] = 255*sub2[row,col,cha]/(sub_max-sub_min)
# sub2 = sub2.astype(np.uint8)


'''
#三种图片的减法,cv2.absdiff(img,background_img)的效果最好
'''
# img_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\480.png'
# img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# cv2.imshow('Img',img)
# background_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\20.png'
# background_img = cv2.imread(background_path,cv2.IMREAD_COLOR)
# cv2.imshow('Background Img',background_img)
#
# sub = cv2.subtract(img,background_img)
# cv2.imshow('subtract Img',sub)
# sub2 = img - background_img
# cv2.imshow('straght sub Img',sub2)
# sub3 = cv2.absdiff(img,background_img)
# cv2.imshow('absdiff Img',sub3)
# print('max1:',sub2.max())
# print('min1:',sub2.min())

#拆分通道测试
# b = cv2.split(img)[0] #B通道
# g = cv2.split(img)[1] #G通道
# r = cv2.split(img)[2] #R通道
# cv2.imshow('b Img',b)
# cv2.imshow('g Img',g)
# cv2.imshow('r Img',r)

'''
#图片的减法,absdiff的运用测试
'''
# #D:\PyCharm\pycharm_project\textureVideoSource\originVideoAndItsFrames\afternoonWithWaterWaterWithColor1027
# img_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\480.png'
# img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# cv2.imshow('Img',img)
#
# background_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\20.png'
# background_img = cv2.imread(background_path,cv2.IMREAD_COLOR)
# cv2.imshow('Background Img',background_img)
#
# absdiff_img = cv2.absdiff(img,background_img)
# cv2.imshow('absdiff Img',absdiff_img)
#
# print('max1:',absdiff_img.max())
# print('min1:',absdiff_img.min())
#
# absdiff_grayImage = cv2.cvtColor(absdiff_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('absdiff gray Img',absdiff_grayImage)

'''
#gabor 特征抽取
'''
# # img_path = 'D:\\PyCharm\\pycharm_project\\lena.jpg'
# #D:\samples_withcolor\pos
# img_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\480.png'
# # img_path = 'D:\\samples_withcolor\\pos\\1.png'
# img = cv2.imread(img_path,cv2.IMREAD_COLOR)
# cv2.imshow('Img',img)
#
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # cv2.imshow('Gray Image',img_gray)
#
# #cv2.getGaborKernel((3,3),2*np.pi,np.pi/2,3,0.5,0,ktype=cv2.CV_32F)
# #cv2.getGaborKernel((3,3),2*np.pi,np.pi/4,3,0.5,0,ktype=cv2.CV_32F)
# #cv2.getGaborKernel((5,5),2*np.pi,np.pi/4,3,0.5,0,ktype=cv2.CV_32F)
# #cv2.getGaborKernel((3,3),np.pi/2,np.pi/2,3,0.5,0,ktype=cv2.CV_32F)
#
# #ksize = (3,3) sigma = 2*np.pi  gamma = 0.5
# # (theta,lambd):(np.pi/2,3.5) (np.pi,3.5) (0,3.5)
# ksize = (3,3)    #gabor核大小,应为奇数,且相同 (9,9)
# sigma = np.pi/2  #默认为2*np.pi
# theta = np.pi/2  #调θ,np.pi/2
# lambd = 3      #调λ,应该2<  <img.shape/5  3.5 3
# gamma = 0.5      #默认不改
# gabor_kernel = cv2.getGaborKernel(ksize,sigma,theta,lambd,gamma,0,ktype=cv2.CV_32F)#((16,16),4.0,np.pi/2,np.pi/4,0.5,0,ktype=cv2.CV_32F)
# # print(gabor_kernel)
# #查看滤波器
# cv2.imshow('gabor_kernel img',gabor_kernel)
#
# gabor_filter = cv2.filter2D(img_gray,-1,gabor_kernel)
# cv2.imshow('Gabor filtered img',gabor_filter)
# # print(gabor_filter[0:2])

'''
#所有像素作为特征进行添加测试
'''
# #D:\samples_withcolor\all_gabor_sixteen_gray
# img_path = 'D:\\samples_withcolor\\all_gabor_sixteen_gray\\1.png'
# gray_image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Gray Image',gray_image)
#
# all_gabor = gray_image.flatten()
# print(all_gabor)
#
# train_hist = np.zeros((1,256))
# train_hist[0] = all_gabor
# print(train_hist)

'''
#在行上合并矩阵的测试
'''
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.array([7,8,9])
# d = np.hstack((a,b,c))
# print(d)

'''
#PCA降维
'''
# '''
# #训练PCA降维,并保存PCA降维模型
# '''
# 创建PCA对象(保留的特征个数)
# pca = PCA(n_components=112)  # 'mle':在满足误差的情况下,自动选择特征个数 112
# 使用数据训练PCA模型
# pca.fit(train_hist)  # 可要可不要
# pca_train_hist = pca.fit_transform(train_hist)  # 等价于pca.fit(train_hist),pca.transform(train_hist)
# print(pca_train_hist[0:2])
# print('after pca, train_hist')
# print(pca_train_hist.shape)
# print(pca.explained_variance_ratio_) #累计方差贡献率
# print(len(pca.explained_variance_ratio_))
# print(pca.explained_variance_ratio_.sum())
# #保存PCA降维模型
# save_file = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_sampledata_pca.pkl'
# save_pca_model(pca, save_file)
# print('save pca model,done.')

# '''
# #加载训练好的PCA模型,对新数据进行降维处理
# '''
# save_file = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_sampledata_pca.pkl'
# with open(save_file,'rb') as infile:
# 	pca_load = pickle.load(infile)['pca_fit']
# #对新的数据data,用已训练好的pca模型进行降维
# train_pca_hist = pca_load.transform(train_hist)
# # print(train_hist)
# print('after pca')
# print(train_pca_hist.shape)
# # print(len(pca_load.explained_variance_ratio_))
# # print(pca_load.explained_variance_ratio_.sum())

'''
#PCA-SVM集成化用到的代码
'''
# '''
# #加载模型进行测试
# '''
# svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_pca_svm_data.xml'#graybgrhsv_lbp_svm_data.xml
# svm_model_load = cv2.ml.SVM_load(svm_model_filepath)
# print('testing...')
# test_result = svm_model_load.predict(train_pca_hist)[1]
#
# differ = test_result == sample_label
# correct_percent = np.count_nonzero(differ)/sample_label.size * 100
# print('Correct_percent:', correct_percent)

# '''
# #加载训练好的SVM模型
# '''
# svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_svm_data.xml'
# svm_model_load = cv2.ml.SVM_load(svm_model_filepath)
#
# '''
# #对训练好的lbp模型使用使用480*480的实际图片进行测试
# '''
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\720.png' #720.png
# test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
# cv2.imshow('Source Image',test_img)
# test_image_clone = test_img.copy()
#
# (winW,winH) = (16,16) #(5,5)
# stepSize = 8 #2 4 8 16
# for (x,y,window) in slidingWindow(test_img,stepSize = stepSize,windowSize=(winW,winH)):
#     if window.shape[0] != winH or window.shape[1] != winW:
#         continue
#     #窗口切片
#     slice = test_img[y:y+winH,x:x+winW]
#     graybgrhsv_lbp_hist = extract_graybgrhsvlbpFeature_from_oneImage(slice)
#     if svm_model_load.predict(graybgrhsv_lbp_hist)[1] == 0:
#         test_image_clone[y:y+winH,x:x+winW] = [0,0,0]
#
# cv2.imshow('After SVM clarified',test_image_clone)


'''
#后续的图像处理1
'''
# open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))  # (16,16)
# after_open = cv2.morphologyEx(test_image_clone, cv2.MORPH_OPEN, open_kernel)
# cv2.imshow('after opening,', after_open)
#
# dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
# after_dilation = cv2.dilate(after_open, dilation_kernel, iterations=1)
# cv2.imshow('after dilation,', after_dilation)
# # 将三通道的黑白图片转换成单通道的黑白图片
# # (abs_after_dilation = cv2.convertScaleAbs(after_dilation) #这个只能转成3通道的uint8,,不能转成1通道的uint8)
# after_dilation = cv2.cvtColor(after_dilation, cv2.COLOR_BGR2GRAY)
#
# print(after_dilation.dtype)
# print(after_dilation.shape)
#
# # 平滑边缘锯齿
# median = cv2.medianBlur(after_dilation, 17)
# cv2.imshow('medianBlur img.', median)
#
# masked = cv2.bitwise_and(test_img, test_img, mask=median)
# cv2.imshow('mask applied to img.', masked)

'''
#后续的图像处理2,SVM纹理做法,已较完善
'''
# close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))  # (16,16)
# after_close = cv2.morphologyEx(test_image_clone, cv2.MORPH_CLOSE, close_kernel)
# cv2.imshow('after close,', after_close)
#
# open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))  # (16,16)
# after_open = cv2.morphologyEx(after_close, cv2.MORPH_OPEN, open_kernel)
# cv2.imshow('after opening,', after_open)
#
# # 将三通道的黑白图片转换成单通道的黑白图片
# # (abs_after_dilation = cv2.convertScaleAbs(after_dilation) #这个只能转成3通道的uint8,,不能转成1通道的uint8)
# after_dilation = cv2.cvtColor(after_open, cv2.COLOR_BGR2GRAY)
#
# contours, hierarchy = cv2.findContours(after_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# hull = cv2.convexHull(cnt, returnPoints=False)
# defects = cv2.convexityDefects(cnt, hull)
#
# corectDefects = np.zeros(after_dilation.shape[0:2], dtype='uint8')
#
# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i, 0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv2.line(corectDefects, start, end, [255, 255, 255], 1)
#     cv2.circle(corectDefects, far, 5, [255, 255, 255], -1)
#
# twoContoursOr = cv2.bitwise_or(after_dilation, corectDefects)
# cv2.imshow('after corect defects,', twoContoursOr)
#
# finalContours, finalHierarchy = cv2.findContours(twoContoursOr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# finalContours.sort(key=len, reverse=True)
# test_img_copy = test_img.copy()
# finalContours_image = cv2.drawContours(test_img_copy, [finalContours[0]], -1, (0, 0, 255), 2)
# finalArea = cv2.contourArea(finalContours[0])
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#
# cv2.putText(test_img_copy, 'Area:' + str(finalArea) + 'pixels', (250, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
# cv2.imshow('Final Contours Image', test_img_copy)
#
# # #平滑边缘锯齿
# # median = cv2.medianBlur(after_dilation,17)
# # cv2.imshow('medianBlur img.', median)
#
# masked = cv2.bitwise_and(test_img, test_img, mask=after_dilation)
# cv2.imshow('mask applied to img.', masked)

'''
#opencv去除高光
'''
# '''
# #图片的自动白平衡算法,灰度世界算法
# @输入参数:
# 	img:彩色图片
# @返回值:
# 	white_balanced_img:白平衡后的图片,彩色图片
# '''
# def white_balance(img):
# 	#拆分b,g,r通道
# 	b, g, r = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
# 	#b,g,r3个通道的平均值k
# 	b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
# 	k = (b_avg + g_avg + r_avg) / 3
# 	#各个通道的增益
# 	kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
# 	#每个通道的每个像素新值
# 	new_b = (b * kb)
# 	new_g = (g * kg)
# 	new_r = (r * kr)
#
# 	#溢出(>255)处理,设为255
# 	for i in range(len(new_b)):
# 		for j in range(len(new_b[0])):
# 			new_b[i][j] = 255 if new_b[i][j] > 255 else new_b[i][j]
# 			new_g[i][j] = 255 if new_g[i][j] > 255 else new_g[i][j]
# 			new_r[i][j] = 255 if new_r[i][j] > 255 else new_r[i][j]
#
# 	# print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
# 	white_balanced_img = np.uint8(np.zeros_like(img))
# 	white_balanced_img[:, :, 0] = new_b
# 	white_balanced_img[:, :, 1] = new_g
# 	white_balanced_img[:, :, 2] = new_r
# 	return white_balanced_img
'''
#去除图片的高光区域
'''
# '''
# #去除图片的高光区域
# @输入参数:
# 	img:彩色图片
# 	highlight_gray:高光区域灰度值的设置,可调整,默认=220
# @返回值:
# 	inpaired_timg:已去除高光区域的彩色图片
# '''
# def remove_highlight(img,highlight_gray = 220):
# 	#转成灰度图片
# 	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	#寻找高光区域
# 	_, biny = cv2.threshold(gray_img, highlight_gray, 255, cv2.THRESH_BINARY)
# 	#对找到的高光区域进行膨胀
# 	dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# 	biny_dilate = cv2.dilate(biny,dilate_kernel,iterations=3)#th1,dilate_kernel,iterations=1,3
# 	#对高光区域进行修复,使用的是Alexandru Telea方法
# 	inpaired_timg = cv2.inpaint(img,biny_dilate,9,cv2.INPAINT_TELEA)
# 	return inpaired_timg

#cv2.illuminationChange方法,未使用
# timg = np.zeros(test_img.shape,np.uint8)
# cv2.illuminationChange(test_img,th2,timg,0.2,0.3)
# cv2.imshow('illuminationChange img', timg)

'''
#白平衡,去高光,整体测试
'''
# #cv2.inpaint方法 1160.png,2700.png
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\760.png'
# test_img1 = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
# cv2.imshow('source img', test_img1)
#
# #白平衡
# test_img = white_balance(test_img1)
# cv2.imshow('balanced img', test_img)
#
#
# gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
# _,biny = cv2.threshold(gray_img,220,255,cv2.THRESH_BINARY) #cv2.threshold(GrayImage,220,255,cv2.THRESH_BINARY)220,230
# cv2.imshow('find highlight img', biny)
#
# dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# biny_dilate = cv2.dilate(biny,dilate_kernel,iterations=3)#th1,dilate_kernel,iterations=1,3
# cv2.imshow('after dilate binary img', biny_dilate)
#
# #cv2.inpaint方法
# inpaired_timg = cv2.inpaint(test_img,biny_dilate,9,cv2.INPAINT_TELEA) #5,9,cv2.INPAINT_NS,cv2.INPAINT_TELEA
# cv2.imshow('inpaired img', inpaired_timg)
#
# inpaired_gaussian_blur = cv2.GaussianBlur(inpaired_timg, (7, 7), 0)
# cv2.imshow('gaussian img', inpaired_gaussian_blur)
#
# inpaired_timg_hsv = cv2.cvtColor(inpaired_gaussian_blur,cv2.COLOR_BGR2HSV)
# cv2.imshow('inpaired hsv img', inpaired_timg_hsv)
#
# # 设定红色的阈值
# rhsv_lower = np.array([0, 40, 56], dtype=np.uint8)
# rhsv_upper = np.array([12, 255, 255], dtype=np.uint8)
# # 根据阈值得到掩模
# hsv_mask = cv2.inRange(inpaired_timg_hsv, rhsv_lower, rhsv_upper)
# cv2.imshow('after hsv color feature filter,', hsv_mask)
#
# # rhsv_lower2 = np.array([0, 40, 56], dtype=np.uint8)
#
#
# # 开运算
# hsv_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_ELLIPSE, (10, 10)
# hsv_open = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, hsv_open_kernel)
# # cv2.imshow('after open,', hsv_open)
# # 膨胀
# hsv_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # cv2.MORPH_RECT,(10,10)
# hsv_open_dilate = cv2.dilate(hsv_open, hsv_dilate_kernel, iterations=1)  # iterations=1
#
# hsv_open_dilate_copy = hsv_open_dilate.copy()
# hsv_contours, hierarchy = cv2.findContours(hsv_open_dilate_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 按轮廓面积对轮廓进行排序,降序排序,冒泡算法
# hsv_lenOfContours = len(hsv_contours)
# for m in range(0, hsv_lenOfContours):
# 	for n in range(0, hsv_lenOfContours - 1 - m):
# 		if cv2.contourArea(hsv_contours[n]) < cv2.contourArea(hsv_contours[n + 1]):
# 			hsv_contours[n], hsv_contours[n + 1] = hsv_contours[n + 1], hsv_contours[n]
#
# # 最大面积的外层轮廓二值图,有凸缺陷的,仅用于显示凸缺陷
# hsv_withDefectsContours = np.zeros(test_img.shape[0:2], dtype='uint8')
# withDefectsContoursImage = cv2.drawContours(hsv_withDefectsContours, [hsv_contours[0]], -1, (255, 255, 255), 1)
#
# '''
# #凸缺陷补偿
# '''
# hsv_cnt = hsv_contours[0]
# hsv_hull = cv2.convexHull(hsv_cnt, returnPoints=False)
# hsv_defects = cv2.convexityDefects(hsv_cnt, hsv_hull)
# # 修正凸缺陷轮廓,二值图
# hsv_corectDefects = np.zeros(test_img.shape[0:2], dtype='uint8')
# for i in range(hsv_defects.shape[0]):
# 	s, e, f, d = hsv_defects[i, 0]
# 	start = tuple(hsv_cnt[s][0])
# 	end = tuple(hsv_cnt[e][0])
# 	far = tuple(hsv_cnt[f][0])
# 	# cv2.line(test_img_copy,start,end,[255,255,255],2)
# 	# cv2.circle(test_img_copy,far,5,[0,0,255],-1)
#
# 	cv2.line(hsv_corectDefects, start, end, [255, 255, 255], 1)
# # cv2.circle(corectDefects,far,5,[255,255,255],-1)
# # 将有凸缺陷的二值图和补偿凸缺陷的二值图进行一个或运算
# hsv_twoContoursOr = cv2.bitwise_or(withDefectsContoursImage, hsv_corectDefects)
# cv2.imshow('Two Contours Or Operated Binary Image', hsv_twoContoursOr)
# hsv_twoContoursOr_copy = hsv_twoContoursOr.copy()
# # 在凸缺陷补偿的或图像寻找最外层的轮廓,cv2.findContours()会直接修改图像，所以需要复制
# hsv_finalContours, finalHierarchy = cv2.findContours(hsv_twoContoursOr_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# hsv_finalContours.sort(key=len, reverse=True)
#
# # 将最外层的轮廓提取出来画在另外一张图上,作为hsv颜色空间最后得到的————>>mask
# hsv_final_contour_out_img = np.zeros(test_img.shape[0:2], dtype='uint8')
# cv2.drawContours(hsv_final_contour_out_img, [hsv_finalContours[0]], -1, (255, 255, 255), 1)
# cv2.imshow('hsv final mask', hsv_final_contour_out_img)
#
# '''
# #可视化的效果
# '''
# # 将轮廓画在原图上进行显示
# hsv_test_img_copy = test_img.copy()
# cv2.drawContours(hsv_test_img_copy, [hsv_finalContours[0]], -1, (0, 0, 255), 2)
# # 显示轮廓面积
# area = cv2.contourArea(hsv_finalContours[0])
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# cv2.putText(hsv_test_img_copy, 'HSV Area:' + str(area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
# cv2.imshow('hsv water area img', hsv_test_img_copy)


'''
#拆分通道
'''
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1300.png'
# test_img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
# cv2.imshow('source img', test_img)
#
# hsv_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv img', hsv_img)

#拆分方法1,cv2.split
# h_img, s_img, v_img = cv2.split(hsv_img)
#拆分方法2,480*480*3的图片组成
# h_img = hsv_img[:,:,0]
# s_img = hsv_img[:,:,1]
# v_img = hsv_img[:,:,2]
#
# cv2.imshow('h img', h_img)
# cv2.imshow('s img', s_img)
# cv2.imshow('v img', v_img)

'''
#图像锐化
'''
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1300.png'
# test_img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
# cv2.imshow('source img', test_img)
#
# source_gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('source_gray_img', source_gray_img)
#
# sharpen_kernel = np.array([[-1,-1,-1],
#                            [-1,9,-1],
#                            [-1,-1,-1]])
# sharpen_img1 = cv2.filter2D(test_img,-1,sharpen_kernel)
# cv2.imshow('sharpen_img1', sharpen_img1)
# gray_img1 = cv2.cvtColor(sharpen_img1,cv2.COLOR_BGR2GRAY)
# cv2.imshow('sharpen gray_img1', gray_img1)
'''
#time模块
'''
# a = time.time()
# b = time.time()
#
# c = b-a
# print(a)
# print(c)
'''
#列表为空的判断
'''
# a = []
# if a:
#     print('1')
# else:
#     print('2')

'''
#创建黑色图片
'''
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\40.png'
# test_img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
# cv2.imshow('source img', test_img)
#
# svm_withDefectsContours = np.zeros(test_img.shape[0:2], dtype='uint8')
# cv2.imshow('svm_withDefectsContoursImage,', svm_withDefectsContours)

# #平滑边缘锯齿
# median = cv2.medianBlur(after_dilation,17)
# cv2.imshow('medianBlur img.', median)

# mask操作,截取感兴趣区域
# masked = cv2.bitwise_and(test_img,test_img,mask=svm_final_contour_out_img)
# cv2.imshow('mask applied to img.', masked)

test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\40.png'
test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('source img', test_img)

open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))  # (cv2.MORPH_RECT/v2.MORPH_ELLIPSE,(16,16))
svm_after_open = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, open_kernel)
cv2.imshow('svm_after_open img', svm_after_open)
print(svm_after_open.shape)


if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()
