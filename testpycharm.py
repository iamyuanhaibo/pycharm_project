# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# #for mat use
# from numpy import *
#
# # img = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# # cv2.imshow('Source Image',img)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# alphas  = mat(zeros((5,1)))
# print(alphas)
#
# classLabels = [-1,1,1,1,-1]
# labelMat = mat(classLabels).transpose()
# print(labelMat)
# n = shape(labelMat)
# print(n)
# inm = [[1,2],[3,4],[5,6]]
# c = shape(inm)
# print(c)


import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import matplotlib.pyplot as plt
import time
import math


# from matplotlib import pyplot as plt
#
# def f(x):
#     return x*x
# a=[1,2,3,4]
# p=map(f,a)
# l = list(p)
# print(type(l))

# x = np.arange(16).reshape(4, 4)
# print(x)
#
# k = [np.hsplit(row,4) for row in np.vsplit(x,4)]
# y = float(k)
#
# print(k)
#
# print(y)
#
# # print(k[3])
# print(len(k))

'''
#倾斜图片的校正例子
'''
# SZ=20
#
# affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
#
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11']/m['mu02']
#     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
#     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
#     return img
#
#
# img = cv2.imread('D:\\data\\0\\4.jpg',0)
# cv2.imshow('Source Image',img)
# print(type(img))
# print(len(img))
# print(img)
#
# #传入的图像需要为单通道的、8位的
# deskewedimg = deskew(img)
# cv2.imshow('Deskewed Image',deskewedimg)


# cells = np.zeros((2,3,4),dtype=np.int32)
# print(cells)
# bin_cells = [1,2],[3,4,5],[6,7,7]
# print(bin_cells)
# print(type(bin_cells))

# [:,np.newaxis]
# responses = np.repeat(np.arange(3),5)
# print(responses)
#
# responses2 = np.repeat(np.arange(3),5)[:,np.newaxis]
# print(responses2)

'''
#视频拆成帧
'''
# cap = cv2.VideoCapture('out.mp4')
#
# i = 0
# while (True):
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.imshow('Video', frame)
#         i = i + 1
#         if i % 20 == 0:
#             cv2.imwrite('D:\\PyCharm\\pycharm_project\\outframes\\' + str(i) + '.png', frame)
#
#         if cv2.waitKey(50) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

# a = 1
# b = 2
#
# c = a and b
# print(c)

'''
验证
#numpy是RGB
#opencv是BGR
'''
# red = np.uint8([[[255,0,0]]])
# hsv_red = cv2.cvtColor(red,cv2.COLOR_RGB2HSV)
# print(hsv_red)

# #cv2.COLOR_RGB2HSV
# green = np.uint8([[[0,255,0]]])
# hsv_green = cv2.cvtColor(green,cv2.COLOR_RGB2HSV)
# print(hsv_green)
#
# red = np.uint8([[[255,0,0]]])
# hsv_red = cv2.cvtColor(red,cv2.COLOR_RGB2HSV)
# print(hsv_red)

'''
# 创建一个开关滑动条
'''
# -*- encoding: utf-8 -*-

# # 定义回调函数，此程序无需回调，所以Pass即可
# def callback(object):
#     pass

#
# img = np.zeros((500, 400, 3), np.uint8)
# cv2.namedWindow('image')
#
# # 创建一个开关滑动条，只有两个值，起开关按钮作用
# switch = '0:OFF\n1:ON'
# cv2.createTrackbar(switch, 'image', 0, 1, callback)
#
# cv2.createTrackbar('R', 'image', 0, 255, callback)
# cv2.createTrackbar('B', 'image', 0, 255, callback)
# cv2.createTrackbar('G', 'image', 0, 255, callback)
#
# while(True):
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#     if cv2.getTrackbarPos(switch, 'image') == 1:
#         img[:] = [b, g, r]
#     else:
#         img[:] = [255, 255, 255]
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#     cv2.imshow('image', img)
# cv2.destroyAllWindows()



'''
#绘制矩形，显示轮廓面积
'''
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\out5frames\\20.png',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
#
# rectang = np.zeros([480,640],dtype='uint8') #img.shape[0:2]
# cv2.imshow('Rectangle',rectang)
# cv2.rectangle(rectang,(25,25),(275,275),(255,255,255),-1)
#
# #cv2.FONT_HERSHEY_COMPLEX_SMALL
# a = 25
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# cv2.putText(img,'Area:'+str(a)+' pixels',(390,460),font,1,(0,0,255),1)
#
#
# # cv2.circle(rectang,(150,150),150,255,-1)
# # cv2.imshow('Rectangle',rectang)
#
# contours, hierarchy = cv2.findContours(rectang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours_image = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)  # img,contours,-1,(255,255,255),1
# cv2.imshow('Contours Image', contours_image)
# # # 轮廓面积
# # area = cv2.contourArea(contours[0])
# # print(area)


# contou = [[1]]

'''
#Python实现冒泡排序
'''
# a = [3,2,5,9,7]
#
# def bubbleSort(arr):
#     n = len(arr)
#     for i in range(0,n-1):
#         for j in range(0,n-1-i):
#             if arr[j] < arr[j+1]:
#                 arr[j],arr[j+1] = arr[j+1],arr[j]
#
# bubbleSort(a)
# print(a)

# a = [[1.2,1.1,2.1],[1.1,2.1,3.1]]
# # a.astype(int)
# int(a)
# print(a)

'''
#2019.10.2
#滑动窗口
'''
# def slidingWindow(image,stepSize,windowSize):
#     for y in range(0,image.shape[0],stepSize):
#         for x in range(0,image.shape[1],stepSize):
#             yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])
#
#
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\lena.jpg',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Img',grayImage)



'''
#2019.09.28
#灰度共生矩阵的四个特征值的计算
'''
# image = np.array([[0, 0, 0, 0, 1, 1],
#                   [0, 0, 0, 0, 1, 1],
#                   [2, 2, 2, 2, 3, 3],
#                   [2, 2, 2, 2, 3, 3]], dtype=np.uint8)
#
#
# print(image.max())
# cv2.imshow('Gray Img',image)

# print(image.shape[0],image.shape[1])

# (winW,winH) = (80,80)
# stepSize = 16
# for (x,y,window) in slidingWindow(grayImage,stepSize = stepSize,windowSize=(winW,winH)):
#     if window.shape[0] != winH or window.shape[1] != winW:
#         continue
#     # print(window.sum())
#     clone = grayImage.copy()
#     cv2.rectangle(clone,(x,y),(x+winW,y+winH),(0,255,0),2)
#     cv2.imshow('Window',clone)
#     cv2.waitKey(50)
    # slice = grayImage[y:y+winH,x+winW]
    # cv2.namedWindow('slidingSlice',0)
    # cv2.imshow('slidingSlice',slice)
    # cv2.waitKey(50)



# result0 = greycomatrix(image, [1], [0],levels = 4,normed=True)
# result45 = greycomatrix(image, [1], [np.pi/4],levels = 4,normed=True)
# result90 = greycomatrix(image, [1], [np.pi/2],levels = 4,normed=True)
# result135 = greycomatrix(image, [1], [3*np.pi/4],levels = 4,normed=True)
#
# result = (result0 + result45 + result90+ result135)/4
# print(result)
#

# a = []
# Energy = greycoprops(result,'energy')
# Contrast = greycoprops(result,'contrast')
# Homogeneity = greycoprops(result,'homogeneity')
# Correlation = greycoprops(result,'correlation')
#
# ASM = greycoprops(result0,'ASM')
# Dissimilarity = greycoprops(result0,'dissimilarity')
#
# # Energy = list(Energy)
# a.append(Energy)
#
# # Contrast = list(Contrast)
# a.append(Contrast)
#
# # Homogeneity = list(Homogeneity)
# a.append(Homogeneity)
#
# # Correlation = list(Correlation)
# a.append(Correlation)
#
# print('Energy的类型:',type(Energy))
# print(a)
# print('\nEnergy:',Energy,'\nContrast:',Contrast,'\nHomogeneity:',Homogeneity,'\nCorrelation:',Correlation)

# print('energy:',Energy,\
#       'contrast:',Contrast,\
#       'homogeneity:',Homogeneity, \
#       'correlation:', Correlation, \
#       'ASM:', ASM,\
#       'dissimilarity:', Dissimilarity)

# mat = [[0,0,0,0,1,1],
#        [0,0,0,0,1,1],
#        [0,0,0,0,1,1],
#        [0,0,0,0,1,1],
#        [2,2,2,2,3,3],
#        [2,2,2,2,3,3]]
#
# matrixCoo = greycomatrix(mat,[1],[0],levels = 4,normed=True,symmetric=False)
#
# print(matrixCoo)

'''
#图片RGB颜色通道的拆分
'''
# (b,g,r) = cv2.split(img)
# cv2.imshow('r Img',b)
# cv2.imshow('g Img',g)
# cv2.imshow('b Img',r)
#
# zeros = np.zeros(img.shape[:2],dtype='uint8')
# cv2.imshow('RED Img',cv2.merge([zeros,zeros,r]))
# cv2.imshow('GREEN Img',cv2.merge([zeros,g,zeros]))
# cv2.imshow('BLUE Img',cv2.merge([b,zeros,zeros]))

# HsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow('HsvImage Img',HsvImage)
#

'''
#图片HSV颜色通道的拆分
'''
# h,s,v = cv2.split(HsvImage)
# # cv2.imshow('h Img',h)
# # cv2.imshow('s Img',s)
# cv2.imshow('v Img',v)
#
# sHistEqu = cv2.equalizeHist(s)
# cv2.imshow('sHistEqu Img',sHistEqu)
#
# merged = cv2.merge([h,s,sHistEqu])
# cv2.imshow('HSV V hist equ Img',merged)

# HsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#
# HistEqu = cv2.equalizeHist(HsvImage[1])
# cv2.imshow('HistEqu Img',HistEqu)


'''
#np.digitize的用法，2019.10.8
'''
# num = np.array([2,6,2,1,3,7,9,8,12,15,-1])
# bins = np.array([0.0,1.0,2.5,4.0,12])
#
# a = len(bins)
# b = num.max()
#
# inds = np.digitize(num,bins)
# print(inds,a,b)





# a = []
# b = (10,8)
# a.append(b)
# c = (12,7)
# a.append(c)
#
# print(a)
#
# print(a[1][1])


# a = 2
# b = 4
#
# if a >= 1 and b <= 10:
#     print('逻辑与操作是and操作')
# else:
#     print('逻辑与操作不是and操作')

# for i in range(5):
#     print(i)

'''
#去除多重括号成单括号，并绘图
'''
# a = [np.array([[1]]),np.array([[1]]),np.array([[1]]),np.array([[1]]),np.array([[1]]),np.array([[1]]),np.array([[1]])]

# a = [[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]]]
#np.flatten不能直接作用于list，只能作用于array，所以需转换一下类型
# a = np.array(a).flatten()
#再把类型转换回来
# a = list(a)
# print(a)
# print(type(a))

# plt.plot(a,'k--')
# plt.show()

# a = list(a)

'''
#累计求和，求平均
'''
# def sy():
#     a = 2
#     return a
# def average():
#     j = 0
#     for i in range(5):
#         j += sy()
#     return (j/5)
#
# print(average())

'''
#限幅滤波测试
'''
# i = 0
# countOutTimes = 2 #最大超时次数
# count = 0 #次数计数
#
# if i <1:
#     a = int(input('Please input first number:'))
#     while a < 0 or a > 10:
#         a = int(input('Wrong number!please input another first number:'))
#         count = count + 1
#         if count == countOutTimes:
#             break
#     count = 0
#     b = int(input('Please input second number:'))
#     while b < 0 or b > 10:
#         b = int(input('Wrong number!please input another second number:'))
#         count = count + 1
#         if count == countOutTimes:
#             break
#     count = 0
#     while abs(a-b) > 3:
#         b = int(input('Wrong number!please input another second number:'))
#         count = count + 1
#         if count == countOutTimes:
#             break
# print('The number input is:',a,b)

'''
# +=测试
'''
# a = 0
# a += 1
#
# print(a)

#显示本地时间
# print(time.asctime(time.localtime(time.time())))

'''
#除法的向上取整
'''
# a = [15,18,33]
# b = 16
# c = []
# #向上取整
# for i in a:
#     c.append(math.ceil(i/b))
# print(c)

'''
#多维列表的向上取整（未成功）
'''
# def mutl(c):
#     return math.ceil(c/3)
# # a = np.ndarray([[1,2,3]
# #                 [4,5,6]
#                 # [7,8,9]])
# # a = [4,5,6]
#
# #列表推导式
# a = [mutl(c) for c in a]
# # c = [[]]
#
# # for i in a[:,:]:
# #     c.append(i/b)
# print(a)

'''
#确定灰度共生矩阵的四个纹理特征阈值
'''
# img = cv2.imread('D:\\PyCharm\\pycharm_project\\6000.png',cv2.IMREAD_COLOR)
# cv2.imshow('Source Img',img)
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Img',grayImage)
#
# #确认感兴趣区域的坐标，画出来就能知道坐标了
# #背景区域
# # cv2.rectangle(img,(0,0),(200,90),(0,0,255),1) #(x,y)
# #积水区域
# cv2.rectangle(img,(150,160),(350,250),(0,0,255),1)
#
# cv2.imshow('ROI Area',img)
# clone = grayImage.copy()
#
# #背景区域
# # ROI = clone[0:90,0:200] #[y1:y2,x1:x2] 水：clone[160:250,150:350]
# #积水区域
# ROI = clone[160:250,150:350]
#
# cv2.imshow('Real ROI Area',ROI)
#
# bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
# #将灰度级降为0-16
# inds = np.digitize(ROI,bins)
#
# #类型转换，inds为int64,将其转为uint8，使符合cv2.imshow的数据类型要求，使之可视化
# indsUint8 = inds.astype(np.uint8)
# cv2.imshow('Inds0',indsUint8)
#
# energyList = []
# #用于灰度共生矩阵的输入参数，灰度最大级数
# maxValue = inds.max()+1
#
# '''
# 计算灰度共生矩阵四个方向（0,45,90,135）的均值灰度共生矩阵
# 注意:inds需要为灰度图像，两像素距离默认为1
# '''
# def getGrayCooMatAverage(grayImage):
#     global maxValue
#     grayCooMat0 = greycomatrix(grayImage, [1], [0], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat45 = greycomatrix(grayImage, [1], [np.pi / 4], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat90 = greycomatrix(grayImage, [1], [np.pi / 2], levels=maxValue, normed=True, symmetric=False)
#     grayCooMat135 = greycomatrix(grayImage, [1], [3 * np.pi / 4], levels=maxValue, normed=True, symmetric=False)
#
#     grayCooMatAverage = (grayCooMat0 + grayCooMat45 + grayCooMat90 + grayCooMat135) / 4
#     return grayCooMatAverage
#
# '''
# #滑动窗口
# #yield的妙用，相当于return，不过是有需要就return，减少内存占用
# '''
# def slidingWindow(image,stepSize,windowSize):
#     for y in range(0,image.shape[0],stepSize):
#         for x in range(0,image.shape[1],stepSize):
#             yield (x,y,image[y:y+windowSize[1],x:x+windowSize[0]])
#
# '''
# #滑动窗口，计算每个滑动窗口的纹理特征值
# '''
# (winW,winH) = (5,5) #(5,5)
# stepSize = 5 #3
#
# for (x,y,window) in slidingWindow(indsUint8,stepSize = stepSize,windowSize=(winW,winH)):
#     if window.shape[0] != winH or window.shape[1] != winW:
#         continue
#     #窗口切片
#     slice = indsUint8[y:y+winH,x:x+winW]
#     # cv2.namedWindow('slidingSlice',0)
#     # cv2.imshow('slidingSlice',slice)
#     grayCooMatAverage = getGrayCooMatAverage(slice)
#     energy = greycoprops(grayCooMatAverage, 'energy')
#     energyList.append(energy)
#
# #去除多重方括号
# energyList = np.array(energyList).flatten()
# energyList = list(energyList)
#
# print(energyList)
# print(len(energyList))
#
#
# #plt.plot(energyList,'k--')
# #plt.plot(energyList,'go--')
# plt.plot(energyList,'y.')
# plt.show()


# for i in range(1,(5+1)):
#     print(i)


if cv2.waitKey(0) & 0xFF ==ord('q'):
    cv2.destroyAllWindows()

# a = [np.vsplit(x,4)]
# print(a)
# print(a[0])

# print(len(a))


# print(k[1])
