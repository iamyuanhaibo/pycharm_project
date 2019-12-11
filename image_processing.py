#Some operations of image pre-processing

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
#Load image
#cv2.IMREAD_COLOR(1),cv2.IMREAD_GRAYSCALE(0),cv2.IMREAD_UNCHANGED(-1)
'''
img = cv2.imread('6.jpg',cv2.IMREAD_COLOR)
cv2.imshow('Source Image',img)

'''
#图像的基本信息
'''
#宽,长,通道数
# print(grayImage.shape[:]) #图像的通道数
inforImage = img.shape
print(inforImage)

#创建图像
#timg = np.zeros(test_img.shape,np.uint8)

#像素总数
cntPixel = img.size
print(cntPixel)

#图像的数据类型
typeImage = img.dtype
print(typeImage)


'''
#Convert to Gray Image
'''
GrayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',GrayImage)

'''
#BINARY IMAGE
#1.genaral threshold binary
#2.adaptive threshold binary
#3.otsu adaptive threshold binary
'''

'''
#1.Genaral threshold binary
'''
#Two return value.threshold value:127.
# ret,thre = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
# cv2.imshow('Binary Image',thre)
'''
#2.Adaptive threshold binary
'''
# #Block Size:11,C:constant
# AdaptiveThresholdaMean = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
# 											   cv2.THRESH_BINARY,11,2)
# cv2.imshow('Adaptive Mean Binary Image',AdaptiveThresholdaMean)
#
# AdaptiveThresholdGaussian = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
# 											   cv2.THRESH_BINARY,11,2)
# cv2.imshow('Adaptive Gaussian Binary Image',AdaptiveThresholdGaussian)
'''
#3.OTSU adaptive threshold binary
'''
# ret,thre1 = cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('OTSU Binary Image',thre1)

'''
#Before edge detect operation,need to remove noise with 5*5 Gaussian filters
#Gradient filter
#high-pass filters to find edge
#1.sobel x
#2.sobel y
#3.laplascian
#4.canny edge detect
'''

'''
#Gaussian Blur
'''
#Gaussian core:(5,5),Standard deviation:0
# cv2.GaussianBlur(GrayImage,(5,5),0)
'''
#1.sobel x
'''
# #cv2.CV_64F,-1:output image depth(CV_8U)
# sobelx = cv2.Sobel(GrayImage,cv2.CV_8U,1,0,ksize = 5)
# cv2.imshow('SobelxEdgeImage',sobelx)
#
# #abs to get uint8 depth
# #sobelx64f = cv2.Sobel(GrayImage,cv2.CV_64F,1,0,ksize=5)
# #abs_sobelx64f = np.absolute(sobelx64f)
# #sobel_8u = np.uint8(abs_sobelx64f)
# #cv2.imshow('Sobelx Edge Image',sobel_8u)
'''
#2.sobel y
'''
# sobely = cv2.Sobel(GrayImage,cv2.CV_8U,0,1,ksize = 5)
# cv2.imshow('Sobely Edge Image',sobely)
#
'''
#3.laplascian
'''
# laplacian = cv2.Laplacian(GrayImage,-1)
# cv2.imshow('Laplacian Edge Image',laplacian)
'''
#4.Canny edge detect
'''
# CannyEdge = cv2.Canny(GrayImage,100,200)
# cv2.imshow('Canny Edge Image',CannyEdge)
###################################################
'''
#Image bluring(image smothing)
#Low-pass filter
#To remove noise and blur image edge
#1.2D convolution
#2.average blur
#3.gaussian blur
#4.median blur
#5.bilateral filter
'''
'''
#1.2D convolution
#average filter(low-pass filter)
#[1 1 ... 1]/25
'''
# kernel = np.ones((5,5),np.float32)/25
# averagefilter = cv2.filter2D(img,-1,kernel)
# cv2.imshow('Convolution Filter Image',averagefilter)
#
'''
#2.average blur
#kernel size:5*5
'''
# averageblur = cv2.blur(img,(5,5))
# cv2.imshow('Average Filter Image',averageblur)
'''
#3.gaussian blur
#gaussian kernel size:5*5
'''
# gaussianblur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow('Gaussian Filter Image',gaussianblur)
#
'''
#4.median blur
#odd number:5
'''
# medianblur = cv2.medianBlur(img,5)
# cv2.imshow('Median Filter Image',medianblur)
'''
#5.bilateral filter
'''
# bilateralblur = cv2.bilateralFilter(img,9,75,75)
# cv2.imshow('Bialateral Filter Image',bilateralblur)
# ~ ###################################################

'''
#Histogram calculation
#1.by opencv
#2.by matplotlib
#3.Histogram equalization
#4.CLAHE equalization
'''
# opencv_hist = cv2.calcHist([img],[0],None,[256],[0,256])
# print(opencv_hist)
# plt.plot(opencv_hist,'g')
#
# plt.hist(GrayImage.ravel(),256,[0,256])
# plt.show()
# #hide tick values on X and Y axis
# plt.xticks([]),plt.yticks([])

# HistEqu = cv2.equalizeHist(GrayImage)
# cv2.imshow('Histogram Equalization Image',HistEqu)
#
# #blocks:8*8
# CLAHE = cv2.createCLAHE(clipLimit = 2.0,tileGridSize = (8,8))
# cl = CLAHE.apply(GrayImage)
# cv2.imshow('Clahe Histogram Equalization Image',cl)
####################################################
'''
#Morphological transformations
#1.Erode operation
#2.Dilation operation
#3.Opening
#4.Closing
#5.Morphological gradient
'''

'''
#1.Erode operation
#To detach two conneted objects
'''
# Kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(GrayImage,Kernel,iterations = 1)
# cv2.imshow('Erode Image',erosion)


'''
#2..Dilation operation
#To connect two parts of an object
'''
#dilation = cv2.dilate(GrayImage,Kernel,iterations = 1)
# cv2.imshow('Dilation Image',dilation)
'''
#3.Opening
#To remove noise around the object
'''
# opening = cv2.morphologyEx(GrayImage,cv2.MORPH_OPEN,Kernel)
# cv2.imshow('Opening Image',opening)
'''
#4.Closing
#To fill holes in the object
'''
# closing = cv2.morphologyEx(GrayImage,cv2.MORPH_CLOSE,Kernel)
# cv2.imshow('Closing Image',closing)
'''
#5.Morphological gradient
#To get the outline of the object
'''
# MorphologicalGradient = cv2.morphologyEx(GrayImage,cv2.MORPH_GRADIENT,Kernel)
# cv2.imshow('Morphological Gradient Image',MorphologicalGradient)
#########################################################

'''
#Find contours,draw contours
'''
#threshold to get a binary image
#ret,thresh = cv2.threshold(GrayImage,127,255,0)
'''
@brief:find contours in a binary image
@param:input binary image
@param:contours search mode
@param:contours approximate method (CHAIN_APPROX_SIMPLE,CHAIN_APPROX_NONE)
@retal:contours
@retal:contours structure
'''
#contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#get area of close-contours
#area = cv2.contourArea(contours[0])
#print(area)
'''
@brief:draw contoure
@param:image
@param:contours
@param:index of contours(note:how many contours)
@param:contours color
@param:contours color thickness
@retal:co-image
'''
# contours_image = cv2.drawContours(img,contours,-1,(255,255,255),1)
# cv2.imshow('binary Image',thresh)
# cv2.imshow('Contours Image',contours_image)
############################################################

if cv2.waitKey(0) & 0xFF ==ord('q'):
	cv2.destroyAllWindows()



