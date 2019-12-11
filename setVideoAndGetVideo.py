#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/19 16:01
# !@Author:yuan
# !@File:setVideoAndGetVideo.py

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
camera=PiCamera()
camera.resolution=(200,200) #分辨率
camera.framerate=20 #帧率

print('saturation:',camera.saturation)
camera.saturation=100 #100 #饱和度
print('after set,saturation:',camera.saturation)

print('brightness:',camera.brightness)
camera.brightness=60 #60 #亮度
print('after set,brightness:',camera.brightness)

print('iso:',camera.iso)
camera.iso=60 #100 #感光度
print('after set,iso:',camera.iso)

#print('shutter_speed:',camera.shutter_speed)
#camera.shutter_speed=5000000 #快门
#print('after set,shutter_speed:',camera.shutter_speed)

print('sharpness:',camera.sharpness)
camera.sharpness=60 #锐度
print('after set,sharpness:',camera.sharpness)

rawCapture=PiRGBArray(camera,size=(200,200))
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture,format='bgr',use_video_port=True):
    image=frame.array
    cv2.imshow('Frame',image)
    rawCapture.truncate(0)
    if cv2.waitKey(1)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
