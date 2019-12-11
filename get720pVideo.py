#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/20 15:01
# !@Author:yuan
# !@File:get720pVideo.py

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

#初始化picamera
camera=PiCamera()
camera.framerate=20 #帧率
camera.resolution=(1280,720) #分辨率
camera.saturation=100 #100 #饱和度
camera.brightness=60 #60 #亮度
camera.iso=60 #100 #感光度
camera.sharpness=60 #锐度

rawCapture=PiRGBArray(camera,size=(1280,720))
#摄像头准备时间
time.sleep(0.1)

forucc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out3.mp4',forucc,20,(1280,720))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    out.write(image)
    cv2.imshow("Frame", image)
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        cv2.destroyAllWindows()
        break