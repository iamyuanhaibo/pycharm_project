#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/12/1 19:24
# !@Author:yuan
# !@File:use_for_raspberry.py

import cv2
import numpy as np

import pickle

from skimage.feature import local_binary_pattern

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
#滑动窗口
#yield的妙用，相当于return，不过是有需要就return，减少内存占用
'''
def slidingWindow(image,stepSize,windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])


'''
#提取归一化后的原始LBP特征
@输入参数：
    grayImage:灰度图片
    radius:lbp半径,默认=1
    n_points:lbp点个数,默认=8
@返回值:
    归一化后的lbp直方图
'''
def originLBP(grayImage,radius = 1,n_points = 8):
    lbp = local_binary_pattern(grayImage, n_points, radius, method='default')
    # 类型转换，lbp为int32，将其转为uint8，为了符合cv2.calcHist()的数据类型要求
    lbpInt = lbp.astype(np.uint8)
    # cv2.imshow('ROILBPImage', lbp)
    # print(lbp[0:3])
    # 参数说明:cv2.calcHist(原图像，灰度图通道0，掩膜图像，BIN的数目，像素值范围)
    # 返回值:opencv_hist是一个256*1的数组(float32)，代表出现[0-255]出现的频次
    opencv_hist = cv2.calcHist([lbpInt], [0], None, [256], [0, 256])
    # print(opencv_hist.dtype)
    #归一化LBP直方图
    opencv_hist /= opencv_hist.sum()
    # print(opencv_hist[0:10])
    return opencv_hist #return opencv_hist

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
#SVM分类后续的图像处理
@输入参数:
	svm_later_img:彩色图片/灰色图片,三通道、单通道均可
@返回值:
	svm_final_mask:经开运算和膨胀后的黑白单通道图片
'''
def svm_later_img_process(svm_later_img):
	# 先开运算再膨胀,矩形核.形态学运算返回的是和原图像一样的属性
	#开运算
	open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(16,16)) #(cv2.MORPH_RECT,(16,16)),(cv2.MORPH_ELLIPSE,(16,16))
	svm_after_open = cv2.morphologyEx(svm_later_img, cv2.MORPH_OPEN, open_kernel)
	#膨胀
	dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #(cv2.MORPH_ELLIPSE, (5, 5)),(cv2.MORPH_ELLIPSE, (16, 16))
	svm_open_dilate = cv2.dilate(svm_after_open, dilate_kernel, iterations=1)
	# 将三通道的黑白图片转换成单通道的黑白图片
	svm_final_mask = cv2.cvtColor(svm_open_dilate, cv2.COLOR_BGR2GRAY)
	return svm_final_mask

if __name__ =='__main__':
	#获得图片
	test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\20.png'
	test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
	cv2.imshow('source img', test_img)

	'''
	#HSV颜色识别
	'''
	#图片白平衡处理
	white_balanced_img = white_balance(test_img)
	#去除高光区域
	remove_highlight_img = remove_highlight(white_balanced_img,highlight_gray = 220)
	#高斯滤波,转成HSV空间才能看出滤波前后的区别
	gaussian_blur = cv2.GaussianBlur(remove_highlight_img,(7,7),0)
	#转成HSV颜色空间
	hsv_img = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)

	'''
	#mask0,红色阈值1:[0, 43, 46]-[10, 255, 255]
	'''
	#设定红色的阈值1
	rhsv_lower0 = np.array([0, 43, 46], dtype=np.uint8)
	rhsv_upper0 = np.array([10, 255, 255], dtype=np.uint8)
	#根据阈值得到掩模mask0
	hsv_mask0 = cv2.inRange(hsv_img, rhsv_lower0, rhsv_upper0)
	#开运算
	hsv_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	hsv_mask0_open = cv2.morphologyEx(hsv_mask0, cv2.MORPH_OPEN, hsv_open_kernel)
	#膨胀
	hsv_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	hsv_mask0_open_dilate = cv2.dilate(hsv_mask0_open,hsv_dilate_kernel,iterations=1)

	'''
	#mask1,红色阈值2:[150, 20, 46]-[180, 255, 255]
	'''
	# 设定红色的阈值2
	rhsv_lower1 = np.array([150, 20, 46], dtype=np.uint8)
	rhsv_upper1 = np.array([180, 255, 255], dtype=np.uint8)
	#根据阈值得到掩模mask1
	hsv_mask1 = cv2.inRange(hsv_img, rhsv_lower1, rhsv_upper1)
	hsv_mask1_open = cv2.morphologyEx(hsv_mask1, cv2.MORPH_OPEN, hsv_open_kernel)
	hsv_mask1_open_dilate = cv2.dilate(hsv_mask1_open, hsv_dilate_kernel, iterations=1)

	#将mask0和mask1进行一个或运算得到hsv final mask
	#hsv final mask
	hsv_final_mask = cv2.bitwise_or(hsv_mask0_open_dilate, hsv_mask1_open_dilate)
	cv2.imshow('hsv final mask', hsv_final_mask)

	'''
	#通过轮廓,得到最大面积轮廓的hsv_mask
	'''
	#寻找轮廓,cv2.findContours()会直接修改图像，所以需要复制
	hsv_open_dilate_copy = hsv_final_mask.copy()
	hsv_contours, hierarchy = cv2.findContours(hsv_open_dilate_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 轮廓会出现为空的情况
	if hsv_contours:
		#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
		hsv_lenOfContours = len(hsv_contours)
		for m in range(0, hsv_lenOfContours):
			for n in range(0, hsv_lenOfContours - 1 - m):
				if cv2.contourArea(hsv_contours[n]) < cv2.contourArea(hsv_contours[n + 1]):
					hsv_contours[n], hsv_contours[n + 1] = hsv_contours[n + 1], hsv_contours[n]

		hsv_contoursMax_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.drawContours(hsv_contoursMax_img, [hsv_contours[0]], -1, (255, 255, 255), 1)
		cv2.imshow('hsv_contoursMax_img', hsv_contoursMax_img)
		'''
		#HSV可视化的效果,可屏蔽
		'''
		# 将轮廓画在原图上进行显示
		hsv_test_img_show = test_img.copy()
		cv2.drawContours(hsv_test_img_show, [hsv_contours[0]], -1, (0, 0, 255), 2)
		# 显示轮廓面积
		hsv_area = cv2.contourArea(hsv_contours[0])
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(hsv_test_img_show, 'HSV Area:' + str(hsv_area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA)
		cv2.imshow('hsv water area img', hsv_test_img_show)

	else:
		hsv_contoursMax_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.imshow('hsv_contoursMax_img', hsv_contoursMax_img)
		'''
		#HSV可视化的效果,可屏蔽
		'''
		# 将轮廓画在原图上进行显示
		hsv_test_img_show = test_img.copy()
		# 显示轮廓面积
		hsv_area = 0
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(hsv_test_img_show, 'HSV Area:' + str(hsv_area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA)
		cv2.imshow('hsv water area img', hsv_test_img_show)

	'''
	#SVM纹理识别
	'''
	#加载PCA模型、SVM模型
	file_save_path = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_samplefeature_pca448_optim.pkl'
	pca_load = load_pca_model(file_save_path)
	svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\graybgrhsv_lbp_pca448_svm_data_optim.xml'
	svm_model_load = cv2.ml.SVM_load(svm_model_filepath)

	'''
	#滑动窗口,依次分类判断,积水为1置为白色[255, 255, 255],非积水为0置为黑色[0,0,0]
	@窗口大小:16*16
	@步长:8
	'''
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
	# cv2.imshow('After SVM clarified,',test_image_clone)
	'''
	#SVM纹理识别后续的图像处理,得到svm final mask
	'''
	svm_final_mask = svm_later_img_process(test_image_clone)
	cv2.imshow('svm final mask', svm_final_mask)

	'''
	#通过轮廓,得到最大面积轮廓的svm_mask
	'''
	#寻找轮廓,cv2.findContours()会直接修改图像，所以需要复制
	svm_final_mask_copy = svm_final_mask.copy()
	svm_contours, svm_hierarchy = cv2.findContours(svm_final_mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 轮廓会出现为空的情况
	#当轮廓不为空
	if svm_contours:
		#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
		svm_lenOfContours = len(svm_contours)
		for m in range(0, svm_lenOfContours):
			for n in range(0, svm_lenOfContours - 1 - m):
				if cv2.contourArea(svm_contours[n]) < cv2.contourArea(svm_contours[n + 1]):
					svm_contours[n], svm_contours[n + 1] = svm_contours[n + 1], svm_contours[n]

		svm_contoursMax_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.drawContours(svm_contoursMax_img, [svm_contours[0]], -1, (255, 255, 255), 1)
		cv2.imshow('svm_contoursMax_img', svm_contoursMax_img)
		'''
		#SVM可视化的效果,可屏蔽
		'''
		# 将轮廓画在原图上进行显示
		svm_test_img_show = test_img.copy()
		cv2.drawContours(svm_test_img_show, [svm_contours[0]], -1, (0, 0, 255), 2)
		# 显示轮廓面积
		svm_area = cv2.contourArea(svm_contours[0])
		cv2.putText(svm_test_img_show0, 'SVM Area:' + str(svm_area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA)
		cv2.imshow('svm water area img', svm_test_img_show)
	else:
		svm_contoursMax_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.imshow('svm_contoursMax_img', svm_contoursMax_img)
		'''
		#SVM可视化的效果,可屏蔽
		'''
		# 将轮廓画在原图上进行显示
		svm_test_img_show = test_img.copy()
		# 显示轮廓面积
		svm_area = 0
		cv2.putText(svm_test_img_show, 'SVM Area:' + str(svm_area) + 'pixels', (100, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA)
		cv2.imshow('svm water area img', svm_test_img_show)

	'''
	#将HSV颜色识别和SVM纹理识别结合
	'''
	svm_hsv_final_mask = cv2.bitwise_or(hsv_contoursMax_img, svm_contoursMax_img)
	cv2.imshow('svm_hsv_final_mask', svm_hsv_final_mask)

	#寻找轮廓,cv2.findContours()会直接修改图像，所以需要复制
	svm_hsv_final_mask_copy = svm_hsv_final_mask.copy()
	svm_hsv_finalContours, svm_hsv_finalHierarchy = cv2.findContours(svm_hsv_final_mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 轮廓会出现为空的情况
	if svm_hsv_finalContours:
		#按轮廓面积对轮廓进行排序,降序排序,冒泡算法
		svm_hsv_lenOfContours = len(svm_hsv_finalContours)
		for m in range(0, svm_hsv_lenOfContours):
			for n in range(0, svm_hsv_lenOfContours - 1 - m):
				if cv2.contourArea(svm_hsv_finalContours[n]) < cv2.contourArea(svm_hsv_finalContours[n + 1]):
					svm_hsv_finalContours[n], svm_hsv_finalContours[n + 1] = svm_hsv_finalContours[n + 1], svm_hsv_finalContours[n]

		#取最大面积的轮廓,面积小的都是噪声轮廓
		svm_hsv_withDefects_contours = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.drawContours(svm_hsv_withDefects_contours, [svm_hsv_finalContours[0]], -1, (255, 255, 255), 1)
		cv2.imshow('svm_hsv_withDefects_contours', svm_hsv_withDefects_contours)

		'''
		#凸缺陷补偿
		'''
		svm_hsv_cnt = svm_hsv_finalContours[0]
		svm_hsv_hull = cv2.convexHull(svm_hsv_cnt, returnPoints=False)
		svm_hsv_defects = cv2.convexityDefects(svm_hsv_cnt, svm_hsv_hull)
		# 修正凸缺陷轮廓,二值图
		svm_hsv_corectDefects = np.zeros(test_img.shape[0:2], dtype='uint8')
		for i in range(svm_hsv_defects.shape[0]):
			s, e, f, d = svm_hsv_defects[i, 0]
			start = tuple(svm_hsv_cnt[s][0])
			end = tuple(svm_hsv_cnt[e][0])
			far = tuple(svm_hsv_cnt[f][0])
			# cv2.line(test_img_copy,start,end,[255,255,255],2)
			# cv2.circle(test_img_copy,far,5,[0,0,255],-1)
			#凸缺陷补偿,只在凸缺陷的地方进行补偿,可能是不闭合的直线
			cv2.line(svm_hsv_corectDefects, start, end, [255, 255, 255], 1)
			# cv2.circle(svm_hsv_corectDefects,far,5,[255,255,255],-1)
		#将有凸缺陷的二值图和补偿凸缺陷的二值图进行一个或运算
		svm_hsv_twoContoursOr = cv2.bitwise_or(svm_hsv_withDefects_contours, svm_hsv_corectDefects)
		cv2.imshow('withDefects_contours Or corectDefects Image', svm_hsv_twoContoursOr)

		svm_hsv_twoContoursOr_copy = svm_hsv_twoContoursOr.copy()
		#在凸缺陷补偿的或图像寻找最外层的轮廓,cv2.findContours()会直接修改图像,所以需要复制
		svm_hsv_final_contour, svm_hsv_final_Hierarchy = cv2.findContours(svm_hsv_twoContoursOr_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		svm_hsv_final_contour.sort(key=len, reverse=True)

		#将最外层的轮廓提取出来画在另外一张图上,作为最后得到的轮廓
		svmhsv_final_contour_out_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.drawContours(svmhsv_final_contour_out_img, [svm_hsv_final_contour[0]], -1, (255, 255, 255), 1)
		cv2.imshow('svmhsv_final_contour_out_img', svmhsv_final_contour_out_img)

		'''
		#HSV-SVM可视化的效果
		'''
		# 将轮廓画在原图上进行显示
		svmhsv_test_img_show = test_img.copy()
		cv2.drawContours(svmhsv_test_img_show, [svm_hsv_final_contour[0]], -1, (0, 0, 255), 2)
		# 显示轮廓面积
		svm_hsv_finalArea = cv2.contourArea(svm_hsv_final_contour[0])
		cv2.putText(svmhsv_test_img_show, 'SVM HSV Area:' + str(svm_hsv_finalArea) + 'pixels', (100, 460), font, 1,(0, 0, 255), 1, cv2.LINE_AA)
		cv2.imshow('svm hsv water area img', svmhsv_test_img_show)
	else:
		svmhsv_final_contour_out_img = np.zeros(test_img.shape[0:2], dtype='uint8')
		cv2.imshow('svmhsv_final_contour_out_img', svmhsv_final_contour_out_img)

		'''
		#HSV-SVM可视化的效果
		'''
		# 将轮廓画在原图上进行显示
		svmhsv_test_img_show = test_img.copy()
		# 显示轮廓面积
		svm_hsv_finalArea = 0
		cv2.putText(svmhsv_test_img_show, 'SVM HSV Area:' + str(svm_hsv_finalArea) + 'pixels', (100, 460), font, 1,(0, 0, 255), 1, cv2.LINE_AA)
		cv2.imshow('svm hsv water area img', svmhsv_test_img_show)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()