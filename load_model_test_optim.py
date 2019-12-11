#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/11/27 15:21
# !@Author:yuan
# !@File:load_model_test_optim.py

import cv2
import numpy as np

from graybgrhsv_lbp_pca_svm import extract_graybgrhsvFeature_saveToVar
from graybgrhsv_lbp_pca_svm import load_pca_model
from LBP_origin_SVM import generate_sample_label
from graybgrhsv_lbp_pca_svm import extract_graybgrhsvlbpFeature_from_oneImage
from LBP_origin_SVM import slidingWindow

from new_trackbarToGetParam import white_balance
from new_trackbarToGetParam import remove_highlight

#增强图片对比度函数,未使用.调用格式:contrast_brightness(img,1.5,0)
def contrast_brightness(img,c,b):
	blank = np.zeros_like(img,img.dtype)
	dst = cv2.addWeighted(img,c,blank,c,b)
	return dst


if __name__ =='__main__':
	#图片源
	#720.png 1480.png
	test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1120.png'
	test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
	cv2.imshow('source img', test_img)

	'''
	#HSV颜色处理
	'''
	print('hsv analysis')
	#图片白平衡处理
	white_balanced_img = white_balance(test_img)
	cv2.imshow('white_balanced_img', white_balanced_img)
	#去除高光区域
	remove_highlight_img = remove_highlight(white_balanced_img,highlight_gray = 220)
	cv2.imshow('remove_highlight_img', remove_highlight_img)

	#高斯滤波,转成HSV空间才能看出滤波前后的区别
	gaussian_blur = cv2.GaussianBlur(remove_highlight_img,(7,7),0)#cv2.GaussianBlur(test_img,(7,7),0)
	# print('after blur')
	cv2.imshow('blur img', gaussian_blur)

	#转成HSV颜色空间
	hsv_img = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
	cv2.imshow('blured hsv_img,', hsv_img)

	'''
	#mask0,红色阈值1:[0, 43, 46]-[10, 255, 255]
	'''
	#设定红色的阈值1
	rhsv_lower0 = np.array([0, 43, 46], dtype=np.uint8)
	rhsv_upper0 = np.array([10, 255, 255], dtype=np.uint8)
	#根据阈值得到掩模mask0
	hsv_mask0 = cv2.inRange(hsv_img, rhsv_lower0, rhsv_upper0)
	# cv2.imshow('after hsv color feature filter got mask0,', hsv_mask0)

	#开运算
	hsv_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #cv2.MORPH_ELLIPSE, (10, 10)
	hsv_mask0_open = cv2.morphologyEx(hsv_mask0, cv2.MORPH_OPEN, hsv_open_kernel)
	# cv2.imshow('mask0,after open,', hsv_mask0_open)
	#膨胀
	hsv_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) #cv2.MORPH_RECT,(10,10)
	hsv_mask0_open_dilate = cv2.dilate(hsv_mask0_open,hsv_dilate_kernel,iterations=1)#iterations=1
	cv2.imshow('mask0,open-dilate,', hsv_mask0_open_dilate)

	'''
	#mask1,红色阈值2:[150, 20, 46]-[180, 255, 255]
	'''
	# 设定红色的阈值2
	rhsv_lower1 = np.array([150, 20, 46], dtype=np.uint8)
	rhsv_upper1 = np.array([180, 255, 255], dtype=np.uint8)
	#根据阈值得到掩模mask1
	hsv_mask1 = cv2.inRange(hsv_img, rhsv_lower1, rhsv_upper1)
	# cv2.imshow('after hsv color feature filter got mask1,', hsv_mask1)
	hsv_mask1_open = cv2.morphologyEx(hsv_mask1, cv2.MORPH_OPEN, hsv_open_kernel)
	# cv2.imshow('mask1,after open,', hsv_mask1_open)
	hsv_mask1_open_dilate = cv2.dilate(hsv_mask1_open,hsv_dilate_kernel,iterations=1)
	cv2.imshow('mask1,open-dilate,', hsv_mask1_open_dilate)

	#hsv final mask
	hsv_final_mask = cv2.bitwise_or(hsv_mask0_open_dilate, hsv_mask1_open_dilate)
	cv2.imshow('hsv_final_mask', hsv_final_mask)

	#寻找轮廓,cv2.findContours()会直接修改图像，所以需要复制
	hsv_open_dilate_copy = hsv_final_mask.copy()
	hsv_contours, hierarchy = cv2.findContours(hsv_open_dilate_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
	hsv_lenOfContours = len(hsv_contours)
	for m in range(0, hsv_lenOfContours):
		for n in range(0, hsv_lenOfContours - 1 - m):
			if cv2.contourArea(hsv_contours[n]) < cv2.contourArea(hsv_contours[n + 1]):
				hsv_contours[n], hsv_contours[n + 1] = hsv_contours[n + 1], hsv_contours[n]

	#最大面积的外层轮廓二值图,有凸缺陷的,仅用于显示凸缺陷
	hsv_withDefectsContours = np.zeros(test_img.shape[0:2], dtype='uint8')
	hsv_withDefectsContoursImage = cv2.drawContours(hsv_withDefectsContours, [hsv_contours[0]], -1, (255, 255, 255), 1)
	cv2.imshow('hsv_withDefectsContoursImage,', hsv_withDefectsContoursImage)

	'''
	#凸缺陷补偿
	'''
	hsv_cnt = hsv_contours[0]
	hsv_hull = cv2.convexHull(hsv_cnt, returnPoints=False)
	hsv_defects = cv2.convexityDefects(hsv_cnt, hsv_hull)
	# 修正凸缺陷轮廓,二值图
	hsv_corectDefects = np.zeros(test_img.shape[0:2], dtype='uint8')
	for i in range(hsv_defects.shape[0]):
		s, e, f, d = hsv_defects[i, 0]
		start = tuple(hsv_cnt[s][0])
		end = tuple(hsv_cnt[e][0])
		far = tuple(hsv_cnt[f][0])
		# cv2.line(test_img_copy,start,end,[255,255,255],2)
		# cv2.circle(test_img_copy,far,5,[0,0,255],-1)

		cv2.line(hsv_corectDefects, start, end, [255, 255, 255], 1)
		# cv2.circle(corectDefects,far,5,[255,255,255],-1)
	#将有凸缺陷的二值图和补偿凸缺陷的二值图进行一个或运算
	hsv_twoContoursOr = cv2.bitwise_or(hsv_withDefectsContours, hsv_corectDefects)
	cv2.imshow('Two Contours Or Operated Binary Image', hsv_twoContoursOr)
	hsv_twoContoursOr_copy = hsv_twoContoursOr.copy()
	#在凸缺陷补偿的或图像寻找最外层的轮廓,cv2.findContours()会直接修改图像,所以需要复制
	hsv_finalContours, finalHierarchy = cv2.findContours(hsv_twoContoursOr_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	hsv_finalContours.sort(key=len, reverse=True)

	#将最外层的轮廓提取出来画在另外一张图上,作为hsv颜色空间最后得到的————>>mask
	hsv_final_contour_out_img = np.zeros(test_img.shape[0:2], dtype='uint8')
	cv2.drawContours(hsv_final_contour_out_img, [hsv_finalContours[0]], -1, (255, 255, 255), 1)
	cv2.imshow('hsv final mask contour', hsv_final_contour_out_img)

	'''
	#可视化的效果
	'''
	#将轮廓画在原图上进行显示
	hsv_test_img_copy = test_img.copy()
	cv2.drawContours(hsv_test_img_copy,[hsv_finalContours[0]],-1,(0,0,255),2)
	#显示轮廓面积
	area = cv2.contourArea(hsv_finalContours[0])
	font = cv2.FONT_HERSHEY_COMPLEX_SMALL
	cv2.putText(hsv_test_img_copy, 'HSV Area:' + str(area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.imshow('hsv water area img', hsv_test_img_copy)


	'''
	#SVM纹理分类
	'''
	# '''
	# #抽取所有测试样本特征,gray-b-g-r-h-s-v lbp特征
	# '''
	# test_sample_path = 'D:\\samples_withcolor\\test_samples\\all_sixteen_step8\\'
	# test_sample_num = 15934
	# test_hist = extract_graybgrhsvFeature_saveToVar(test_sample_path, test_sample_num)
	# print(test_hist.shape)
	#
	# '''
	# #所有样本特征对应标记
	# '''
	# pos_sample_num = 3913
	# neg_sample_num = 12021
	# # 正负样本lbp特征对应的标记,负样本在前,正样本在后
	# sample_label = generate_sample_label(pos_sample_num, neg_sample_num, pos_label=1, neg_label=0, negpos=1)

	# 加载模型进行测试
	#加载PCA模型
	print('svm analysis')
	file_save_path = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_samplefeature_pca448_optim.pkl'
	pca_load = load_pca_model(file_save_path)
	#加载SVM模型
	#2019年12月06日,优化模型:graybgrhsv_lbp_pca1792_svm_data_optim.xml
	svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_pca448_svm_data_optim.xml'
	svm_model_load = cv2.ml.SVM_load(svm_model_filepath)
	test_image_clone = test_img.copy()

	(winW,winH) = (16,16) #(5,5)
	stepSize = 8 #2 4 8 16
	for (x,y,window) in slidingWindow(test_img,stepSize = stepSize,windowSize=(winW,winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		#窗口切片
		slice = test_img[y:y+winH,x:x+winW]
		slice_feature = extract_graybgrhsvlbpFeature_from_oneImage(slice)
		pca_slice_feature = pca_load.transform(slice_feature)
		if int(svm_model_load.predict(pca_slice_feature)[1]) == 0:
			test_image_clone[y:y+winH,x:x+winW] = [0,0,0]
		else:
			test_image_clone[y:y + winH, x:x + winW] = [255, 255, 255]
	cv2.imshow('After SVM clarified,',test_image_clone)

	'''
	#SVM分类后续的图像处理
	'''
	#先开运算再膨胀,矩形核
	open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(16,16)) #(cv2.MORPH_RECT/v2.MORPH_ELLIPSE,(16,16)),
	svm_after_open = cv2.morphologyEx(test_image_clone, cv2.MORPH_OPEN, open_kernel)
	cv2.imshow('after open,', svm_after_open)
	dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #(cv2.MORPH_ELLIPSE,(16,16))
	svm_open_dilate = cv2.dilate(svm_after_open,dilate_kernel,iterations=1)#iterations=1
	cv2.imshow('open-dilate,', svm_open_dilate)

	#将三通道的黑白图片转换成单通道的黑白图片
	#(abs_after_dilation = cv2.convertScaleAbs(after_dilation) #这个只能转成3通道的uint8,,不能转成1通道的uint8)
	svm_open_dilate = cv2.cvtColor(svm_open_dilate, cv2.COLOR_BGR2GRAY)

	#寻找轮廓
	svm_contours, svm_hierarchy = cv2.findContours(svm_open_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#轮廓会出现为空的情况
	# print(svm_contours)
	#若轮廓列表不为空,即有轮廓的话
	if svm_contours:
	#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
		svm_lenOfContours = len(svm_contours)
		for m in range(0, svm_lenOfContours):
			for n in range(0, svm_lenOfContours - 1 - m):
				if cv2.contourArea(svm_contours[n]) < cv2.contourArea(svm_contours[n + 1]):
					svm_contours[n], svm_contours[n + 1] = svm_contours[n + 1], svm_contours[n]

		#最大面积的外层轮廓二值图,带凸缺陷
		svm_withDefectsContours = np.zeros(test_image_clone.shape[0:2], dtype='uint8')
		cv2.drawContours(svm_withDefectsContours, [svm_contours[0]], -1, (255, 255, 255), 1)
		cv2.imshow('svm_withDefectsContoursImage,', svm_withDefectsContours)

		'''
		#凸缺陷补偿
		'''
		svm_cnt = svm_contours[0]
		svm_hull = cv2.convexHull(svm_cnt, returnPoints=False)
		svm_defects = cv2.convexityDefects(svm_cnt, svm_hull)
		svm_corectDefects = np.zeros(test_image_clone.shape[0:2], dtype='uint8')
		for i in range(svm_defects.shape[0]):
			s, e, f, d = svm_defects[i, 0]
			start = tuple(svm_cnt[s][0])
			end = tuple(svm_cnt[e][0])
			far = tuple(svm_cnt[f][0])
			cv2.line(svm_corectDefects, start, end, [255, 255, 255], 1)
			# cv2.circle(corectDefects, far, 5, [255, 255, 255], -1)
		#将有凸缺陷的二值图和补偿凸缺陷的二值图进行一个或运算
		svm_twoContoursOr = cv2.bitwise_or(svm_withDefectsContours, svm_corectDefects)
		cv2.imshow('svm after corect defects,', svm_twoContoursOr)

		svm_finalContours, svm_finalHierarchy = cv2.findContours(svm_twoContoursOr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		svm_finalContours.sort(key=len, reverse=True)

		#将最外层的轮廓提取出来画在另外一张图上,作为svm最后得到的————>>mask
		global svm_final_contour_out_img
		svm_final_contour_out_img = np.zeros(test_image_clone.shape[0:2], dtype='uint8')
		cv2.drawContours(svm_final_contour_out_img, [svm_finalContours[0]], -1, (255, 255, 255), 1)
		cv2.imshow('svm final mask', svm_final_contour_out_img)
		'''
		#可视化的效果
		'''
		#将轮廓画在原图上进行显示
		test_img_copy1 = test_img.copy()
		cv2.drawContours(test_img_copy1, [svm_finalContours[0]], -1, (0, 0, 255), 2) #(test_img_copy1, [svm_finalContours[0]], -1, (0, 0, 255), 2)
		# 显示轮廓面积
		svm_finalArea = cv2.contourArea(svm_finalContours[0])
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(test_img_copy1, 'SVM Area:' + str(svm_finalArea) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
		cv2.imshow('svm water area img', test_img_copy1)
	else:
		# global svm_final_contour_out_img
		svm_final_contour_out_img = np.zeros(test_image_clone.shape[0:2], dtype='uint8')
		cv2.imshow('svm final mask', svm_final_contour_out_img)
		test_img_copy1 = test_img.copy()
		svm_finalArea = 0
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(test_img_copy1, 'SVM Area:' + str(svm_finalArea) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
		cv2.imshow('svm water area img', test_img_copy1)

	'''
	#将HSV的颜色区分和SVM的分类结合
	'''
	svm_hsv_final_mask = cv2.bitwise_or(svm_final_contour_out_img, hsv_final_contour_out_img)
	cv2.imshow('svm hsv or mask,', svm_hsv_final_mask)
	svm_hsv_finalContours, svm_hsv_finalHierarchy = cv2.findContours(svm_hsv_final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# svm_hsv_finalContours.sort(key=len, reverse=True)

	#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
	svm_hsv_lenOfContours = len(svm_hsv_finalContours)
	for m in range(0, svm_hsv_lenOfContours):
		for n in range(0, svm_hsv_lenOfContours - 1 - m):
			if cv2.contourArea(svm_hsv_finalContours[n]) < cv2.contourArea(svm_hsv_finalContours[n + 1]):
				svm_hsv_finalContours[n], svm_hsv_finalContours[n + 1] = svm_hsv_finalContours[n + 1], svm_hsv_finalContours[n]

	#取最大面积的轮廓,面积小的都是噪声轮廓
	svm_hsv_final_contour_out_img = np.zeros(test_image_clone.shape[0:2], dtype='uint8')
	cv2.drawContours(svm_hsv_final_contour_out_img, [svm_hsv_finalContours[0]], -1, (255, 255, 255), 1)
	cv2.imshow('svm hsv final contour,', svm_hsv_final_contour_out_img)

	'''
	#可视化效果
	'''
	#将轮廓画在原图上
	test_img_copy2 = test_img.copy()
	cv2.drawContours(test_img_copy2, [svm_hsv_finalContours[0]], -1, (0, 0, 255), 2)
	#将计算面积画在原图上
	svm_hsv_finalArea = cv2.contourArea(svm_hsv_finalContours[0])
	font = cv2.FONT_HERSHEY_COMPLEX_SMALL
	cv2.putText(test_img_copy2, 'SVM HSV Area:' + str(svm_hsv_finalArea) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.imshow('svm hsv water area img', test_img_copy2)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()