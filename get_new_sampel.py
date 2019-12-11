#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/14 16:45
# !@Author:yuan
# !@File:get_new_sample.py

'''
#将正样本减去背景后作为训练的正样本
 负样本保持不变进行训练,得出的模型正确率57%左右,效果不佳
'''


import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

from cutPictureToWindows import save_image_to_files
from LBP_origin_SVM import extractFeature_saveToVar
from LBP_origin_SVM import generate_sample_label
from LBP_origin_SVM import extract_lbpFeature_from_oneImage
from LBP_origin_SVM import slidingWindow



#设置矩阵在pycharm中全显示，不自动换行,防止写入txt数据格式不对
np.set_printoptions(linewidth=500000)
np.set_printoptions(threshold=np.inf)

# cv2.imshow('Background Img',background_img)

#保存正样本与背景图之差
# pos_path = 'D:\\absdiff\\pos\\'
# pos_num = 75
# for j in range(1, pos_num + 1):
# 	pos_image = cv2.imread(pos_path + str(j) + '.png', cv2.IMREAD_COLOR)
# 	# cv2.imshow('Source Img', pos_image)
# 	absdiff_img = cv2.absdiff(pos_image, background_img)
# 	#D:\absdiff\pos_absdiff
# 	save_path = 'D:\\absdiff\pos_absdiff\\'
# 	cv2.imwrite(save_path+str(j)+'.png', absdiff_img)
# print('save done.')

#保存样本至文件夹
# pos_sample_num = 207
# pos_sample_path = 'D:\\absdiff\\pos_absdiff\\imageclipper\\'
# pos_save_file_path = 'D:\\samples_withcolor\\pos_absdiff_sixteen\\'
# save_image_to_files(pos_sample_path,pos_sample_num,pos_save_file_path,winW=16,winH=16,stepSize=8)

# '''
# #所有样本特征
# '''
# # D:\samples_withcolor\ano_negpos_sixteen
# all_sample_path = 'D:\\samples_withcolor\\ano_negpos_sixteen\\'
# all_sample_num = 70090
# # 正负样本lbp特征,负样本：1-8423、正样本：8424-14577
# train_hist = extractFeature_saveToVar(all_sample_path, all_sample_num)
#
# '''
# #所有样本特征对应标记
# '''
# pos_sample_num = 31246
# neg_sample_num = 38844
# # 正负样本lbp特征对应的标记,负样本在前,正样本在后
# sample_label = generate_sample_label(pos_sample_num, neg_sample_num, pos_label=1, neg_label=0, negpos=1)
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
# print('SVM training...')
# svm.train(train_hist, cv2.ml.ROW_SAMPLE, sample_label)
# svm.save('originlbp_withcolor_ano_svm_data.xml')
# print('SVM train,done.')


'''
#加载训练好的SVM模型
'''
# svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\originlbp_withcolor_ano_svm_data.xml'
# svm_model_load = cv2.ml.SVM_load(svm_model_filepath)
'''
#对训练好的lbp模型使用16*16的小图片进行简单的测试
'''
# # 测试样本
# test_pos_sample_num = 31246 #6154
# test_neg_sample_num = 38844 #8423
# test_sample_path = 'D:\\samples_withcolor\\ano_negpos_sixteen\\'
#
# #测试样本的lbp特征,后送入SVM分类模型
# test_pos_sample_lbpfeature = extractFeature_saveToVar(test_sample_path,test_pos_sample_num+test_neg_sample_num)
# #测试样本的正确标记
# test_label = generate_sample_label(test_pos_sample_num,test_neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1)
#
# #测试样本的分类结果
# test_result = svm_model_load.predict(test_pos_sample_lbpfeature)[1]
# # a = np.array(test_result).flatten()
# # print(a)
#
# #differ:两矩阵的不同为0,相同为1
# differ = test_result == test_label
# #正确率
# correct_percent = np.count_nonzero(differ)/test_label.size*100
# print(correct_percent)

'''
#对训练好的lbp模型使用480*480的实际图片进行测试
'''
# test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\720.png' #1760.png
# test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
# cv2.imshow('Source Image',test_img)
# test_image_clone = test_img.copy()
#
# test_img_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('Gray Image',test_img_gray)
# test_img_gray_clone = test_img_gray.copy()
#
# (winW,winH) = (16,16) #(5,5)
# stepSize = 8 #2 4 8 16
# for (x,y,window) in slidingWindow(test_img_gray_clone,stepSize = stepSize,windowSize=(winW,winH)):
#     if window.shape[0] != winH or window.shape[1] != winW:
#         continue
#     #窗口切片
#     slice = test_img_gray_clone[y:y+winH,x:x+winW]
#     origin_lbp_hist = extract_lbpFeature_from_oneImage(slice)
#     if int(svm_model_load.predict(origin_lbp_hist)[1]) == 0:
#         test_image_clone[y:y+winH,x:x+winW] = [0,0,0]
#
# cv2.imshow('After SVM clarified',test_image_clone)

if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()
