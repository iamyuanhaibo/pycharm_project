#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/11 19:56
# !@Author:yuan
# !@File:videoGet.py
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#读取摄像头自己的参数，图像长、宽、帧速度等
#具体参数对应见<<OpenCV官方教程中文版(For Python)>> P26
# cap = cv2.VideoCapture(0)
# print(cap.get(3),cap.get(4))

forucc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out2.mp4',forucc,20,(640,480))

while cap.isOpened()  :
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        # cv2.namedWindow('SouVideo',cv2.WINDOW_GUI_NORMAL)
        # cv2.resizeWindow('SouVideo',320,320)
        cv2.imshow('SouVideo', frame)

        #cv2.imshow('GrayVideo', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('camera capture interrupted!')
            break
    else:
        print('camera is not ready!')
        break

cap.release()
out.release()
cv2.destroyAllWindows()