#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/6 15:35
# !@Author:yuan
# !@File:svmToRecongnizeNumber.py

# import cv2
# import numpy as np
#
# img = cv2.imread('digits.png',1)
# # cv2.imshow('Source Image',img)
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# '''
# digits.png的切割
# 切割成20×20的，0-9各500张，总计5000张
# 并各自放到对应的目录，图片编号为1-5000（可自行修改）
# 要求：要先在D盘下，新建一个data的文件夹，然后分别新建0-9的子目录
# '''
# # k=0
# # piturename = 0
# # for j in range(1,50+1):
# #
# #     for i in range(100):
# #         #gray[y,x]，而不是gray[x,y]
# #         roi = gray[20*(j-1):20*(j-1)+20,20*i:(20*i+20)]
# #         # cv2.imshow('ROI Image',roi)
# #         piturename = 'D:\\data\\'+str(k)+'\\'
# #         piturename = piturename+str((j-1)*100+i+1)+'.jpg'
# #         cv2.imwrite(piturename,roi)
# #     if j % 5 == 0:
# #         k = k + 1
#
# cells = [np.hsplit(row,100)for row in np.vsplit(gray,50)]

import cv2
import numpy as np


SZ=20  #图像大小，20×20，所以只定义一个变量
bin_n = 16 # Number of bins

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

#图像仿射，校正图像（将歪的图像摆正）
#mu02为moments的y方向的二阶中心距，表倾斜程度
#skew猜测为偏移的方向角度
#M为2×3的转换矩阵，偏移多少转回多少
#warpAffine为图像几何变换中的仿射变换
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy) #幅值，方向
    bins = np.int32(bin_n*ang/(2*np.pi))    # 将（0,2π）的方向转换成（0,16）
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # .ravel()将多维数组转成一维数组(flatten()也可以，但返回的是源数据的副本，ravel不会，也就是说不会进行copy操作，更好些)
    #.squeeze()也可以多转一，但是得是（n,1）的数据形式，.reshape(-1)好像也可以多转一
    # zip()组成元组，方向、幅值一一对应
    #np.bincount返回（0,16）的频率
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    #将64维特征向量平铺成一行量
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

img = cv2.imread('digits.png',0)

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     Now training      ########################
#这里进行了更改
#python3，map返回的是迭代器，而不是list
deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]

#np.reshape 平铺，将（m,n）矩阵转换成（1，n）
trainData = np.float32(hogdata).reshape(-1,64)
# print(trainData)

#转成（250，1）矩阵的0-9
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
print(responses)
print(type(responses))
print(responses.dtype)

#这里进行了更改
# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)


#cv2.ml.ROW_SAMPLE表示需要以行来存储
#训练数据为行向量，对应标签为列向量
# svm.train(trainData,cv2.ml.ROW_SAMPLE,responses)
# svm.save('svm_data.xml')





#
# ######     Now testing      ########################
#
# deskewed = [list(map(deskew,row)) for row in test_cells]
# hogdata = [list(map(hog,row)) for row in deskewed]
# testData = np.float32(hogdata).reshape(-1,bin_n*4)
# result = svm.predict(testData)[1]
#
# #######   Check Accuracy   ########################
# mask = result==responses
# correct = np.count_nonzero(mask)
# print (correct*100.0/result.size)

'''
#测试用例
'''
# test = cv2.imread('3016.jpg',1)
# testGray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
# testDeskewed = deskew(testGray)
# cv2.imshow('Deskewed Image',testDeskewed)
#
# testHogdata = hog(testDeskewed)
# testData = np.float32(testHogdata).reshape(-1,64)
# testNumber = svm.predict(testData)[1]
# print(testNumber)
# print(type(testNumber))




if cv2.waitKey(0) & 0xFF ==ord('q'):
	cv2.destroyAllWindows()