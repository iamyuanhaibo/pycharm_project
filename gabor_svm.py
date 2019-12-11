#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/18 17:18
# !@Author:yuan
# !@File:gabor_svm.py

import cv2
import numpy as np

from LBP_origin_SVM import slidingWindow
from LBP_origin_SVM import generate_sample_label
from LBP_origin_SVM import extractFeature_saveToVar

def save_gamboredimage_to_files(sample_path,sample_num,save_file_path,winW=16,winH=16,stepSize=16):
	i = 1
	gabor_kernel = cv2.getGaborKernel((3, 3), np.pi / 2, np.pi / 2, 3, 0.5, 0, ktype=cv2.CV_32F)
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
			gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
			gabor_filter = cv2.filter2D(gray, -1, gabor_kernel)
			cv2.imwrite(save_file_path + str(i) + '.png',gabor_filter)
			cv2.imshow('slidingSlice',slice)
			cv2.waitKey(10)
			i += 1
	print('Save 16 times 16 gabor image to files,done.')


def extract_gaborFeature_saveToVar(sample_path,sample_num):
	train_hist = np.zeros((sample_num,256))
	for j in range(1,sample_num+1):
		grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
		gabor_feature = grayImage.flatten()
		train_hist[j-1] = gabor_feature
	return np.float32(train_hist)



if __name__ =='__main__':
	'''
	#正样本
	'''
	# 样本数
	pos_sample_num = 102
	# 原负样本路径:D:\samples_withcolor\pos
	pos_sample_path = 'D:\\samples_withcolor\\pos\\'
	# 保存的负样本路径:D:\samples_withcolor\pos_sixteen_gabor
	pos_save_file_path = 'D:\\samples_withcolor\\pos_sixteen_gabor\\'
	# save_gamboredimage_to_files(pos_sample_path,pos_sample_num,pos_save_file_path,winW=16,winH=16,stepSize=6)

	'''
	#负样本
	'''
	neg_sample_num = 253
	neg_sample_path = 'D:\\samples_withcolor\\neg\\'
	neg_save_file_path = 'D:\\samples_withcolor\\neg_sixteen_gabor\\'
	# save_gamboredimage_to_files(neg_sample_path,neg_sample_num,neg_save_file_path,winW=16,winH=16,stepSize=16)

	'''
	#所有样本gabor特征,这里取的是所有像素点
	'''
	all_train_sample_path = 'D:\\samples_withcolor\\all_gabor_sixteen_gray\\'
	all_train_sample_num = 72395
	# gabor滤波后,所有像素点做为特征
	# train_hist = extract_gaborFeature_saveToVar(all_train_sample_path,all_train_sample_num) #47.76711%
	# gabor滤波后,再提取LBP做为特征
	# train_hist = extractFeature_saveToVar(all_train_sample_path,all_train_sample_num) #39.75412%

	'''
	#所有样本特征对应标记
	'''
	pos_sample_num = 33551
	neg_sample_num = 38844
	# 正负样本lbp特征对应的标记,负样本在前,正样本在后
	sample_label = generate_sample_label(pos_sample_num, neg_sample_num, pos_label=1, neg_label=0, negpos=1)

	'''
	#SVM参数设置
	'''
	svm = cv2.ml.SVM_create()
	svm.setKernel(cv2.ml.SVM_RBF)
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setC(1)
	svm.setGamma(7)
	'''
	#训练并保存SVM模型
	'''
	print('SVM training...')
	svm.train(train_hist, cv2.ml.ROW_SAMPLE, sample_label)
	# svm.save('gabor_svm_data.xml')
	print('SVM train,done.')
	print('testing...')
	test_result = svm.predict(train_hist)[1]
	differ = test_result == sample_label
	correct_percent = np.count_nonzero(differ)/sample_label.size*100
	print('Correct_percent:', correct_percent)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
