#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/8 14:56
# !@Author:yuan
# !@File:uniformLBP_SVM.py

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

np.set_printoptions(linewidth=5000000)
np.set_printoptions(threshold=np.inf)

'''
#提取灰度图片的旋转不变uniform LBP特征归一化后的直方图
@输入参数:
    grayImage:灰度图片
    radius:LBP半径大小,默认=1
    n_points:LBP点的个数,n_points = 8*radius,默认=8
@返回值:
    旋转不变uniform LBP特征归一化后的直方图,[[],[],...,[]]的格式,floa型
'''
def uniformLBP(grayImage,radius = 1,n_points = 8):
    lbp = local_binary_pattern(grayImage, n_points, radius, method='uniform')
    # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
    lbpInt = lbp.astype(np.uint8)
    # cv2.imshow('uniformLBPImage', lbp)
    # print(lbp[0:3])
    # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
    # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
    opencv_hist = cv2.calcHist([lbpInt], [0], None, [10], [0, 10])#cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
	#归一化
    opencv_hist /= opencv_hist.sum()
    # print(opencv_hist)
    return opencv_hist

def extract_UniformLBPFeature_saveToVar(sample_path,sample_num):
    train_hist = np.zeros((sample_num,10))
    for j in range(1,sample_num+1):
        grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        opencv_hist = uniformLBP(grayImage,radius = 1,n_points = 8)
        opencv_hist = opencv_hist.flatten()
        #只保留小数点后四位
        # opencv_hist = np.around(opencv_hist,4)
        # print(opencv_hist)
        # print(len(opencv_hist))
        train_hist[j-1] = opencv_hist
    return np.float32(train_hist)

def generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1):
    sample_label = []
    if negpos == 1:
        for i in range(neg_sample_num):
            sample_label.append(neg_label)
        for j in range(pos_sample_num):
            sample_label.append(pos_label)
    else:
        for i in range(pos_sample_num):
            sample_label.append(pos_label)
        for j in range(neg_sample_num):
            sample_label.append(neg_label)
    sample_label = np.array(sample_label)
    sample_label = sample_label[:,np.newaxis]
    return sample_label

#从单个16*16的灰度图片抽取uniformLBP特征
def extract_grayImage_uniformLBP(grayImage_path):
    train_hist = np.zeros((1, 10))
    grayImage = cv2.imread(grayImage_path, cv2.IMREAD_GRAYSCALE)
    opencv_hist = uniformLBP(grayImage, radius=1, n_points=8)
    opencv_hist = opencv_hist.flatten()
    train_hist[0] = opencv_hist
    return np.float32(train_hist)


'''
#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
'''
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])

'''
#从16*16的灰度图片中提取原始lbp特征
@输入参数:
    16*16的灰度图片
@返回值:
    归一化后的lbp特征向量直方图,行向量1*256,float32
'''
def extract_lbpFeature_from_oneImage(grayImage):
    train_hist = []
    opencv_hist = uniformLBP(grayImage,radius = 1,n_points = 8)
    opencv_hist = opencv_hist.flatten()
    # 只保留小数点后四位
    opencv_hist = np.around(opencv_hist, 4)
    #将[]转为[[],[],[]...,[]]
    train_hist.append(opencv_hist)
    train_hist = np.array(train_hist)
    return np.float32(train_hist)





'''
#训练模型
'''
# '''
# #训练特征及对应标记,特征为负样本在前,正样本在后,标记需要对应
# '''
# all_sample_num = 72395 #44097
# all_sample_path = 'D:\\samples_withcolor\\all_posneg\\'
# train_hist = extract_UniformLBPFeature_saveToVar(all_sample_path,all_sample_num)
# # print(train_hist[0:1])
#
# neg_sample_num = 38844 #38844
# pos_sample_num = 33551 #5253
# train_label = generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 0)
# # print(train_label[38840:38850])
#
# '''
# #SVM参数设置
# '''
# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_RBF)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)
# '''
# #训练并保存SVM模型
# '''
# print('uniform LBP SVM training...')
# svm.train(train_hist,cv2.ml.ROW_SAMPLE,train_label)
# svm.save('posneg_uniformlbp_svm_data.xml')
# print('uniform LBP SVM train,done.')


#加载训练好的SVM模型
svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\posneg_uniformlbp_svm_data.xml'
load_svm_model = cv2.ml.SVM_load(svm_model_filepath)
'''
#对训练好的lbp模型使用16*16的小图片进行简单的测试
'''
# #测试1:单张16*16的测试图片
# grayImage_path = 'D:\\samples_withcolor\\all_sixteen_gray_step16\\40000.png'
# grayImage_path_uniformLBP_feature = extract_grayImage_uniformLBP(grayImage_path)
# predict_result = load_svm_model.predict(grayImage_path_uniformLBP_feature)[1]
# print(predict_result)

#测试2:多张16*16测试样本,测试模型预测正确率
# test_pos_sample_num = 33551 #5253
# test_neg_sample_num = 0 #38844
# test_sample_path = 'D:\\samples_withcolor\\all_posneg\\'
#
# #测试样本的lbp特征,后送入SVM分类模型
# test_pos_sample_lbpfeature = extract_UniformLBPFeature_saveToVar(test_sample_path,(test_pos_sample_num+test_neg_sample_num))
# #测试样本的正确标记
# test_label = generate_sample_label(test_pos_sample_num,test_neg_sample_num,pos_label = 1,neg_label = 0,negpos = 0)
# #测试样本的分类结果
# test_result = load_svm_model.predict(test_pos_sample_lbpfeature)[1]
# #differ:两矩阵的不同为0,相同为1
# differ = test_result == test_label
# #正确率
# correct_percent = np.count_nonzero(differ)/test_label.size*100
# print(correct_percent)

'''
#对训练好的lbp模型使用使用480*480的实际图片进行测试
'''
test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\2500.png' #1760.png
test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
cv2.imshow('Source Image',test_img)
test_image_clone = test_img.copy()

test_img_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',test_img_gray)
test_img_gray_clone = test_img_gray.copy()

(winW,winH) = (16,16) #(5,5)
stepSize = 8 #2 4 8 16
for (x,y,window) in slidingWindow(test_img_gray_clone,stepSize = stepSize,windowSize=(winW,winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    #窗口切片
    slice = test_img_gray_clone[y:y+winH,x:x+winW]
    origin_lbp_hist = extract_lbpFeature_from_oneImage(slice)
    if int(load_svm_model.predict(origin_lbp_hist)[1]) == 0:
        test_image_clone[y:y+winH,x:x+winW] = [0,0,0]

cv2.imshow('After SVM clarified',test_image_clone)




if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()
