#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/11 22:48
# !@Author:yuan
# !@File:cutVideoToFrames.py
import cv2
import numpy as np

cap = cv2.VideoCapture('D:\\PyCharm\\pycharm_project\\textureVideoSource\\afternoonWithWaterWaterWithColor1027.mp4')
i = 0
while (True):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Video', frame)
        i = i + 1
        if i % 20 == 0:
            cv2.imwrite('D:\\PyCharm\\pycharm_project\\textureVideoSource\\afternoonWithWaterWaterWithColor1027\\' + str(i) + '.png', frame)
        #20*50 = 1000ms抽一帧
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break


#D:\PyCharm\pycharm_project\textureVideoSource\afternoonWithoutWaterFrames
cap.release()
cv2.destroyAllWindows()

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
