#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/7/25 21:29
# !@Author:yuan
# !@File:watersheld_use.py



import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Watershold algorithm
'''

img = cv2.imread('water_coins.jpg',1)
cv2.imshow('Source Image',img)

GrayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',GrayImage)
#OTSU to get a binary image
ret,thresh = cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('OTSU Image',thresh)

#opening to remove noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#sure background area
sure_backgr = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow('sure_backgr Image',sure_backgr)

#distance transfer to get foreground
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.imshow('dist_transform Image',dist_transform)

ret,sure_foregr = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_foregr = np.uint8(sure_foregr)

#finding unknown region
unknown = cv2.subtract(sure_backgr,sure_foregr)
cv2.imshow('sure_foregr Image',sure_foregr)

#marker labelling
ret,markers = cv2.connectedComponents(sure_foregr)

#add one to all labels so sure background is not 0,but 1
markers = markers + 1

#mark the unknown region with 0
markers[unknown == 255] =0

#use wawtershed algorithm
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,255]

cv2.imshow('Edge Image',img)

if cv2.waitKey(0) & 0xFF ==ord('q'):
	cv2.destroyAllWindows()
