#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/31 21:47
# !@Author:yuan
# !@File:extractFeaturesFromImages.py

'''
#使用SVM训练原始LBP分类模型,并进行简单测试
@准备正负样本
@抽取原始LBP特征并做好标记
@SVM参数设定
@训练得出模型,保存模型文件
@加载模型进行测试:小图片测试和实际图片测试
'''

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops,local_binary_pattern
import matplotlib.pyplot as plt
import time
import math

#设置矩阵在pycharm中全显示，不自动换行,防止写入txt数据格式不对
np.set_printoptions(linewidth=5000000)
np.set_printoptions(threshold=np.inf)

'''
注意,cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
读取标志：若读灰度图用cv2.IMREAD_COLOR，读出来也是3通道
          读灰度图应用cv2.IMREAD_GRAYSCALE
          读完整图片，包括alpha通道：cv2.IMREAD_UNCHANGED
'''

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
#提取lbp特征,并写入至txt文件,sample_num个[0.2,0.32,...,]256个元素
@输入参数:
    sample_path:待提取灰度图片(16*16)的路径
    sample_num:待提取灰度图片的总数
    txt_path:最后生成的要写入的txt文件路径及文件名称
@返回值：
    无,最后打印完成提示信息
'''
def extractFeature_saveToTxt(sample_path,sample_num,txt_path):
    for j in range(1,sample_num+1):
        grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Source Img',grayImage)
        # print(grayImage.shape[:]) #图像的通道数
        opencv_hist = originLBP(grayImage,radius = 1,n_points = 8)
        opencv_hist = opencv_hist.flatten()
        #只保留小数点后四位
        opencv_hist = np.around(opencv_hist,4)
        # print(opencv_hist)
        #查看维度
        # print(opencv_hist.shape)

        #写入文件，防止覆盖，写入属性w改为a
        with open(txt_path,'a') as file_object:
            file_object.write(str(opencv_hist) + '\n')#+ '\n'
    # 提示提取特征完成
    print('Extract lbp features from the samples,done!')

'''
#写入正负样本标记至txt文件,格式如[1. 1. 1. ... 0. 0. 0. ...]
@输入参数:
    pos_sample_num:正样本数
    neg_sample_num:负样本数
    all_sample_label_path:最后生成的所有正负样本标记保存的路径及文件名称
    pos_label:正样本标记,默认=1
    neg_label:负样本标记,默认=0
    negpos:标记对应的特征的正、负样本的顺序,取值1或0(非1),默认 = 1:负样本在前，正样本在后
           以便对应正负样本特征的顺序
@返回值:
    无,最后打印写入标记完成提示信息
'''
def save_all_sample_labels(pos_sample_num,neg_sample_num,all_sample_label_path,pos_label = 1,neg_label = 0,negpos = 1):
    sample_label = np.zeros((pos_sample_num + neg_sample_num))
    if negpos == 1:
        for i in range(pos_sample_num + neg_sample_num):
            if i <= (neg_sample_num-1):
                sample_label[i] = neg_label
            else:
                sample_label[i] = pos_label
    else:
        for i in range(pos_sample_num + neg_sample_num):
            if i <= (pos_sample_num-1):
                sample_label[i] = neg_label
            else:
                sample_label[i] = pos_label
    with open(all_sample_label_path, 'a') as file_object:
        file_object.write(str(sample_label))  # file_object.write(str(i)+'\n')
        file_object.close()

    print('Write sample labels,done!')

'''
#提取lbp特征,sample_num个[0.2,0.32,...,]256个元素
@输入参数:
    sample_path:待提取灰度图片(16*16)的路径
    sample_num:待提取灰度图片的总数
@返回值：
    返回图片的lbp特征矩阵
'''
def extractFeature_saveToVar(sample_path,sample_num):
    train_hist = np.zeros((sample_num,256))
    for j in range(1,sample_num+1):
        grayImage = cv2.imread(sample_path+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        opencv_hist = originLBP(grayImage,radius = 1,n_points = 8)
        opencv_hist = opencv_hist.flatten()
        #只保留小数点后四位
        opencv_hist = np.around(opencv_hist,4)
        # print(opencv_hist)
        # print(len(opencv_hist))
        train_hist[j-1] = opencv_hist
    return np.float32(train_hist)

'''
#获取对应正负样本特征的标记,格式如[1. 1. 1. ... 0. 0. 0. ...]
@输入参数:
    pos_sample_num:正样本数
    neg_sample_num:负样本数
    pos_label:正样本标记,默认=1
    neg_label:负样本标记,默认=0
    negpos:标记对应的特征的正、负样本的顺序,取值1或0(非1),默认 = 1:负样本在前，正样本在后
           以便对应正负样本特征的顺序    
@返回值:
    对应正负样本特征的标记
'''
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
    opencv_hist = originLBP(grayImage, radius=1, n_points=8)
    opencv_hist = opencv_hist.flatten()
    # 只保留小数点后四位
    opencv_hist = np.around(opencv_hist, 4)
    #将[]转为[[],[],[]...,[]]
    train_hist.append(opencv_hist)
    train_hist = np.array(train_hist)
    return np.float32(train_hist)

if __name__ =='__main__':
    '''
    #将样本特征及对应标记写入TXT文件所需信息
    '''
    # '''
    # #正样本信息
    # '''
    # pos_sample_path = 'D:\\samples\\pos_sixteen_gray\\'
    # pos_sample_num = 6154
    # pos_txt_path = 'D:\\samples\\pos_lbp_features.txt'
    # '''
    # #负样本信息
    # '''
    # neg_sample_path = 'D:\\samples\\neg_sixteen_gray\\'
    # neg_sample_num = 8423
    # neg_txt_path = 'D:\\samples\\neg_lbp_features.txt'
    #
    # '''
    # #所有样本信息
    # '''
    # all_sample_num = 14577
    # all_sample_path = 'D:\\samples\\all_sixteen_gray\\'
    # all_txt_path = 'D:\\samples\\all_lbp_features.txt'
    # '''
    # #样本标记信息
    # '''
    # pos_sample_num = 6154
    # neg_sample_num = 8423
    # all_sample_label_path = 'D:\\samples\\all_sample_label.txt'
    #提取lbp特征,并写入至txt文件
    # extractFeature_saveToTxt(all_sample_path,all_sample_num,all_txt_path)
    #写入正/负样本的特征对应的标记至txt文件
    # save_all_sample_labels(pos_sample_num,neg_sample_num,all_sample_label_path,pos_label = 1,neg_label = 0,negpos = 1)

    '''
    #训练模型使用
    '''
    #D:\samples_withcolor\all_sixteen_gray_step16
    all_sample_path = 'D:\\samples_withcolor\\all_sixteen_gray_step16\\'
    all_sample_num = 72395
    #正负样本lbp特征,负样本：1-8423、正样本：8424-14577
    train_hist = extractFeature_saveToVar(all_sample_path,all_sample_num)

    pos_sample_num = 33551
    neg_sample_num = 38844
    #正负样本lbp特征对应的标记,负样本在前,正样本在后
    sample_label = generate_sample_label(pos_sample_num,neg_sample_num,pos_label = 1,neg_label = 0,negpos = 1)

    '''
    #SVM参数设置
    '''
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1) #1
    svm.setGamma(7) #7
    print('SVM training...')
    svm.train(train_hist, cv2.ml.ROW_SAMPLE, sample_label)
    print('SVM train,done.')
    print('testing...')
    test_result = svm.predict(train_hist)[1]
    differ = test_result == sample_label
    correct_percent = np.count_nonzero(differ) / sample_label.size * 100
    print('Correct_percent:', correct_percent)

    # '''
    # #SVM参数设置,变参数尝试寻找最优C和Gamma值
    # '''
    # for kernel in [cv2.ml.SVM_LINEAR,cv2.ml.SVM_SIGMOID]:#cv2.ml.SVM_POLY
    #     for c in range(1,10+1,2):
    #         for gamma in range(1,10+1,2):
    #             svm = cv2.ml.SVM_create()
    #             svm.setKernel(kernel)#cv2.ml.SVM_RBF
    #             svm.setType(cv2.ml.SVM_C_SVC)
    #             # svm.setDegree() #cv2.ml.SVM_POLY核使用
    #             svm.setC(c) #2.67
    #             svm.setGamma(gamma) #5.383
    #             '''
    #             #训练并保存SVM模型
    #             '''
    #             print('SVM training...')
    #             svm.train(train_hist,cv2.ml.ROW_SAMPLE,sample_label)
    #             # svm.save('originlbp_withcolor_svm_data.xml')
    #             print('SVM train,done.')
    #             print('testing...')
    #             test_result = svm.predict(train_hist)[1]
    #             differ = test_result == sample_label
    #             correct_percent = np.count_nonzero(differ)/sample_label.size*100
    #             print('Kernel:',kernel,'C:',c,'Gamma:',gamma,'Correct_percent:',correct_percent)
    #             # print('C:1','Gamma:7','Correct_percent:',correct_percent)

    '''
    #svm.trainAuto
    '''
    # svm = cv2.ml.SVM_create()
    # print('SVM training...')
    # a = time.time()
    # svm.trainAuto(train_hist,cv2.ml.ROW_SAMPLE,sample_label)
    # b = time.time()
    # test_result = svm.predict(train_hist)[1]
    # differ = test_result == sample_label
    # correct_percent = np.count_nonzero(differ) / sample_label.size * 100
    # print('SVM,training done.','Train time:',(b-a),' s')
    # print('Correct_percent:', correct_percent)


    '''
    #加载训练好的SVM模型
    '''
    # svm_model_filepath = 'D:\\PyCharm\\pycharm_project\\originlbp_withcolor_svm_data.xml'
    # svm_model_load = cv2.ml.SVM_load(svm_model_filepath)

    '''
    #对训练好的lbp模型使用16*16的小图片进行简单的测试
    '''
    #测试样本
    # test_pos_sample_num = 33551 #6154
    # test_neg_sample_num = 38844 #8423
    # test_sample_path = 'D:\\samples_withcolor\\all_sixteen_gray_step16\\'
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
    #对训练好的lbp模型使用使用480*480的实际图片进行测试
    '''
    # test_image_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\2500.png' #1760.png
    # test_img = cv2.imread(test_image_path,cv2.IMREAD_COLOR)
    # cv2.imshow('Source Image',test_img)
    # test_image_clone = test_img.copy()
    #
    # test_img_gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
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
    # cv2.imwrite('D:\\samples_withcolor\\sourceImage_SVMClarifiedImage\\2500_originLBP.png',test_image_clone)


    #得到分类后的图片后,可再做一些图片处理以得到更好的结果
    # test_image_clone_gray = cv2.cvtColor(test_image_clone,cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(0) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
