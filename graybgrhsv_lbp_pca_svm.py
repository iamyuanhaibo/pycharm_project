#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/19 20:10
# !@Author:yuan
# !@File:graybgrhsv_lbp_pca_svm.py

import cv2
import numpy as np

from LBP_origin_SVM import originLBP
from LBP_origin_SVM import generate_sample_label
from LBP_origin_SVM import slidingWindow

from sklearn.decomposition import PCA #scikit-learn库
import pickle
import time

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
		gray_hist = np.around(gray_hist, 4)#保留4位小数
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

		#hsv图像
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		#拆分h、s、v通道
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
    img:16*16的彩色图片
@返回值:
    train_hist:归一化后的lbp特征向量直方图,行向量1*(256*7),float32
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

'''
#将训练好的PCA模型保存下来,并返回经PCA降维后的样本特征数据
@输入参数:
	train_sample:训练PCA降维模型的样本,格式如[[],[],[]...[]]
	save_feature_nums:降维后保留的主成分数,推荐值112
	file_save_path:保存PCA模型文件的路径文件名,格式如:'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_sampledata_pca.pkl'
@返回值:
	pca_train_hist,经PCA降维后的样本特征数据
'''
def save_pca_model_to_files(train_sample,file_save_path,save_feature_num = 112):
	#创建PCA对象(保留的特征个数)
	pca = PCA(n_components=save_feature_num) #'mle':在满足误差的情况下,自动选择特征个数 112
	#使用数据训练PCA模型
	pca.fit(train_sample)
	pca_train_hist = pca.fit_transform(train_sample)
	#保存PCA降维模型
	# with open(file_save_path,'wb') as outfile:
	# 	pickle.dump({'pca_fit':pca},outfile)
	return pca_train_hist

'''
#加载训练好的PCA模型文件
@输入参数:
	file_save_path:训练好的PCA模型文件路径文件名称,格式如'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_sampledata_pca.pkl'
@返回值:
	pca_load_model:加载的PCA模型
'''
def load_pca_model(file_save_path):
	with open(file_save_path, 'rb') as infile:
		pca_load_model = pickle.load(infile)['pca_fit']
	return pca_load_model

'''
#训练并保存SVM模型文件
@输入参数:
	train_sample:训练SVM模型的样本
	sample_label:样本对应的标记
	svm_save_filename:训练好的SVM模型文件保留名,格式如'graybgrhsv_lbp_pca_svm_data.xml'
	kernel:SVM核,默认cv2.ml.SVM_RBF
	c:SVM模型中的C值,默认2.67
	gamma:SVM模型中的gamma值,默认5.383
@返回值:
	svm:保存SVM模型文件,并返回训练好的SVM模型
'''
def svm_train_save(train_sample,sample_label,svm_save_filename,kernel = cv2.ml.SVM_RBF,c = 2.67,gamma = 5.383):
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(kernel)
	svm.setC(c)
	svm.setGamma(gamma)
	svm.train(train_sample, cv2.ml.ROW_SAMPLE, sample_label)
	# svm.save(svm_save_filename)
	return svm

if __name__ =='__main__':
	'''
	#抽取所有样本特征,gray-b-g-r-h-s-v lbp特征
	'''
	#D:\samples_withcolor\sample_optim\all_sixteen
	all_sample_path = 'D:\\samples_withcolor\\sample_optim\\all_sixteen\\'
	all_sample_num = 70983 #72395
	train_hist = extract_graybgrhsvFeature_saveToVar(all_sample_path, all_sample_num)
	print(train_hist.shape)
	'''
	#所有样本特征对应标记
	'''
	pos_sample_num = 21821
	neg_sample_num = 49162
	# 正负样本lbp特征对应的标记,负样本在前,正样本在后
	sample_label = generate_sample_label(pos_sample_num, neg_sample_num, pos_label=1, neg_label=0, negpos=1)

	#测试样本
	test_sample_path = 'D:\\samples_withcolor\\test_samples\\all_sixteen_step8\\'
	test_sample_num = 15934
	test_hist = extract_graybgrhsvFeature_saveToVar(test_sample_path, test_sample_num)
	test_pos_sample_num = 3913
	test_neg_sample_num = 12021
	test_label = generate_sample_label(test_pos_sample_num, test_neg_sample_num, pos_label=1, neg_label=0, negpos=1)


	'''
	#训练PCA降维,并保存PCA降维模型.(函数方式好像有点问题)
	'''
	# save_file = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_samplefeature_pca.pkl'
	# pca_train_hist = save_pca_model_to_files(train_hist, save_file, save_feature_num=112)

	# 创建PCA对象(保留的特征个数)
	pca = PCA(n_components=448)  # 'mle':在满足误差的情况下,自动选择特征个数 112
	# 使用数据训练PCA模型
	pca.fit(train_hist)  # 可要可不要
	pca_train_hist = pca.fit_transform(train_hist)  # 等价于pca.fit(train_hist),pca.transform(train_hist)
	# 保存PCA模型文件
	# save_file = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_samplefeature_pca448_optim.pkl'
	# with open(save_file,'wb') as outfile:
	# 	pickle.dump({'pca_fit':pca},outfile)
	# print('after pca,train hist')
	# print(pca_train_hist.shape)

	'''
	#将经PCA降维后的特征样本送入SVM进行训练,保存并返回已训练好的SVM模型.(函数方式好像有点问题)
	'''
	# svm_save_filename = 'graybgrhsv_lbp_pca_svm_data.xml'
	# svm = svm_train_save(pca_train_hist, sample_label, svm_save_filename, kernel=cv2.ml.SVM_RBF, c=2.67, gamma=5.383)
	'''
	#SVM参数设置
	'''
	time_1 = time.time()
	svm = cv2.ml.SVM_create()
	svm.setKernel(cv2.ml.SVM_RBF) #cv2.ml.SVM_RBF
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setC(7)  # 1  7
	svm.setGamma(9)  # 7 9
	print('SVM training...')
	svm.train(pca_train_hist, cv2.ml.ROW_SAMPLE, sample_label)
	time_2 = time.time()
	print('train time:',(time_2-time_1))
	#保存SVM训练模型
	# svm_save_filename = 'graybgrhsv_lbp_pca448_svm_data_optim.xml'
	# svm.save(svm_save_filename)
	print('SVM train,done.')

	'''
	#穷举法,寻找SVM参数最优C和Gamma值
	'''
	# print('testing...')
	# for c in range(5,15+1,2):
	# 	for gamma in range(7,17+1,2):
	# 		svm = cv2.ml.SVM_create()
	# 		svm.setKernel(cv2.ml.SVM_RBF)#cv2.ml.SVM_RBF
	# 		svm.setType(cv2.ml.SVM_C_SVC)
	# 		# svm.setDegree() #cv2.ml.SVM_POLY核使用
	# 		svm.setC(c) #2.67
	# 		svm.setGamma(gamma) #5.383
	# 		'''
	# 		#训练并保存SVM模型
	# 		'''
	# 		print('SVM training...')
	# 		svm.train(train_hist,cv2.ml.ROW_SAMPLE,sample_label)
	# 		# svm.save('originlbp_withcolor_svm_data.xml')
	# 		print('SVM train,done.')
	# 		print('testing...')
	# 		test_result = svm.predict(test_hist)[1]
	# 		differ = test_result == test_label
	# 		correct_percent = np.count_nonzero(differ)/test_label.size*100
	# 		print('C:',c,'Gamma:',gamma,'Correct_percent:',correct_percent)
	# 		# print('C:1','Gamma:7','Correct_percent:',correct_percent)
	# print('test,done.')

	'''
	#pca后,进行简单训练样本的预测测试
	'''
	# print('train sample testing...')
	# test_result = svm.predict(pca_train_hist)[1]
	# differ = test_result == sample_label
	# correct_percent = 100*np.count_nonzero(differ)/sample_label.size
	# print('train sample correct_percent:', correct_percent)

	'''
	#pca后,进行简单测试样本的预测测试
	'''
	# print('test sample testing...')
	# pca_test_hist = pca.transform(test_hist)
	# test_result1 = svm.predict(pca_test_hist)[1]
	# differ1 = test_result1 == test_label
	# correct_percent1 = 100*np.count_nonzero(differ1)/test_label.size
	# print('test sample correct_percent:', correct_percent1)

	'''
	#进行简单训练样本的预测测试
	'''
	# print('train sample testing...')
	# test_result = svm.predict(train_hist)[1]
	# differ = test_result == sample_label
	# correct_percent = 100*np.count_nonzero(differ)/sample_label.size
	# print('train sample correct_percent:', correct_percent)

	'''
	#进行简单测试样本的预测测试
	'''
	# print('test sample testing...')
	# test_result1 = svm.predict(test_hist)[1]
	# differ1 = test_result1 == test_label
	# correct_percent1 = 100*np.count_nonzero(differ1)/test_label.size
	# print('test sample correct_percent:', correct_percent1)


	'''
	#加载模型进行测试,训练样本测试,测试样本测试
	'''
	print('load model test')
	file_save_path = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_samplefeature_pca448_optim.pkl'
	pca_load = load_pca_model(file_save_path)

	svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_pca448_svm_data_optim.xml'
	svm_model_load = cv2.ml.SVM_load(svm_model_filepath)

	#训练样本测试
	print('train sample testing...')
	# 使用训练好的PCA模型,对数据进行降维,transform方法
	pca_train_hist1 = pca_load.transform(train_hist)
	print('after pca,train hist')
	print(pca_train_hist1.shape)
	test_result = svm_model_load.predict(pca_train_hist1)[1]
	differ = test_result == sample_label
	correct_percent = 100*np.count_nonzero(differ)/sample_label.size
	print('train sample correct_percent:', correct_percent)
	#测试样本测试
	print('test sample testing...')
	pca_test_hist = pca_load.transform(test_hist)
	test_result1 = svm.predict(pca_test_hist)[1]
	differ1 = test_result1 == test_label
	correct_percent1 = 100*np.count_nonzero(differ1)/test_label.size
	print('test sample correct_percent:', correct_percent1)






	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
