#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/19 11:12
# !@Author:yuan
# !@File:rgb_gray_hsi.py

import cv2
import numpy as np

from LBP_origin_SVM import originLBP
from LBP_origin_SVM import generate_sample_label
from LBP_origin_SVM import slidingWindow

'''
#提取gray-b-g-r-h-s-v lbp特征,sample_num个[0.2,0.32,...,]256*7个元素
@输入参数:
    sample_path:待提取彩色图片(16*16)的路径
    sample_num:待提取灰度图片的总数
@返回值：
    返回图片的gray-b-g-r-h-s-v lbp特征矩阵
'''
def extract_graybgrhsvFeature_saveToVar(sample_path,sample_num):
	train_hist = np.zeros((sample_num, 256*7))
	for j in range(1, sample_num + 1):
		img = cv2.imread(sample_path+str(j)+'.png', cv2.IMREAD_COLOR)
		# 灰度图片
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		gray_hist = originLBP(gray_img, radius=1, n_points=8)
		gray_hist = gray_hist.flatten()
		gray_hist = np.around(gray_hist, 4)
		# print(gray_hist)

		# 拆分通道,b、g、r通道
		b_img, g_img, r_img = cv2.split(img)

		b_img_hist = originLBP(b_img, radius=1, n_points=8)
		b_img_hist = b_img_hist.flatten()
		b_img_hist = np.around(b_img_hist, 4)

		g_img_hist = originLBP(g_img, radius=1, n_points=8)
		g_img_hist = g_img_hist.flatten()
		g_img_hist = np.around(g_img_hist, 4)

		r_img_hist = originLBP(r_img, radius=1, n_points=8)
		r_img_hist = r_img_hist.flatten()
		r_img_hist = np.around(r_img_hist, 4)

		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h_img, s_img, v_img = cv2.split(hsv_img)

		h_img_hist = originLBP(h_img, radius=1, n_points=8)
		h_img_hist = h_img_hist.flatten()
		h_img_hist = np.around(h_img_hist, 4)

		s_img_hist = originLBP(s_img, radius=1, n_points=8)
		s_img_hist = s_img_hist.flatten()
		s_img_hist = np.around(s_img_hist, 4)

		v_img_hist = originLBP(v_img, radius=1, n_points=8)
		v_img_hist = v_img_hist.flatten()
		v_img_hist = np.around(v_img_hist, 4)
		#在行上将矩阵合并
		train_hist[j-1] = np.hstack((gray_hist,b_img_hist,g_img_hist,r_img_hist,h_img_hist,s_img_hist,v_img_hist))
	train_hist = np.around(train_hist, 4)
	return np.float32(train_hist)

'''
#从16*16的彩色图片中提取gray-b-g-r-h-s-v lbp特征
@输入参数:
    16*16的彩色图片
@返回值:
    归一化后的lbp特征向量直方图,行向量1*(256*7),float32
'''
def extract_graybgrhsvlbpFeature_from_oneImage(img):
	train_hist = np.zeros((1, 256 * 7))
	# 灰度图片
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	gray_hist = originLBP(gray_img, radius=1, n_points=8)
	gray_hist = gray_hist.flatten()
	gray_hist = np.around(gray_hist, 4)
	# print(gray_hist)

	# 拆分通道,b、g、r通道
	b_img, g_img, r_img = cv2.split(img)

	b_img_hist = originLBP(b_img, radius=1, n_points=8)
	b_img_hist = b_img_hist.flatten()
	b_img_hist = np.around(b_img_hist, 4)

	g_img_hist = originLBP(g_img, radius=1, n_points=8)
	g_img_hist = g_img_hist.flatten()
	g_img_hist = np.around(g_img_hist, 4)

	r_img_hist = originLBP(r_img, radius=1, n_points=8)
	r_img_hist = r_img_hist.flatten()
	r_img_hist = np.around(r_img_hist, 4)

	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h_img, s_img, v_img = cv2.split(hsv_img)

	h_img_hist = originLBP(h_img, radius=1, n_points=8)
	h_img_hist = h_img_hist.flatten()
	h_img_hist = np.around(h_img_hist, 4)

	s_img_hist = originLBP(s_img, radius=1, n_points=8)
	s_img_hist = s_img_hist.flatten()
	s_img_hist = np.around(s_img_hist, 4)

	v_img_hist = originLBP(v_img, radius=1, n_points=8)
	v_img_hist = v_img_hist.flatten()
	v_img_hist = np.around(v_img_hist, 4)
	# 在行上将矩阵合并
	train_hist[0] = np.hstack((gray_hist, b_img_hist, g_img_hist, r_img_hist, h_img_hist, s_img_hist, v_img_hist))
	train_hist = np.around(train_hist, 4)

	return np.float32(train_hist)

if __name__ =='__main__':
	# '''
	# #所有样本特征,gray-b-g-r-h-s-v lbp特征
	# '''
	# all_sample_path = 'D:\\samples_withcolor\\all_sixteen_color\\'
	# all_sample_num = 72395
	# train_hist = extract_graybgrhsvFeature_saveToVar(all_sample_path, all_sample_num)
	#
	# '''
	# #所有样本特征对应标记
	# '''
	# pos_sample_num = 33551
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
	# svm.save('graybgrhsv_lbp_svm_data.xml')
	# print('SVM train,done.')
	#
	# print('testing...')
	# test_result = svm.predict(train_hist)[1]
	# differ = test_result == sample_label
	# correct_percent = np.count_nonzero(differ)/sample_label.size * 100
	# print('Correct_percent:', correct_percent)
	'''
	#加载训练好的SVM模型
	'''
	svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_svm_data.xml'
	svm_model_load = cv2.ml.SVM_load(svm_model_filepath)

	'''
	#对训练好的lbp模型使用使用480*480的实际图片进行测试
	'''
	test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\720.png' #720.png
	test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
	cv2.imshow('Source Image',test_img)
	test_image_clone = test_img.copy()

	(winW,winH) = (16,16) #(5,5)
	stepSize = 8 #2 4 8 16
	for (x,y,window) in slidingWindow(test_img,stepSize = stepSize,windowSize=(winW,winH)):
	    if window.shape[0] != winH or window.shape[1] != winW:
	        continue
	    #窗口切片
	    #窗口切片
	    slice = test_img[y:y+winH,x:x+winW]
	    graybgrhsv_lbp_hist = extract_graybgrhsvlbpFeature_from_oneImage(slice)
	    if svm_model_load.predict(graybgrhsv_lbp_hist)[1] == 0:
	        test_image_clone[y:y+winH,x:x+winW] = [0,0,0]

	cv2.imshow('After SVM clarified',test_image_clone)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()