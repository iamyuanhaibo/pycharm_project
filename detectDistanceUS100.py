#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/22 11:20
# !@Author:yuan
# !@File:detectDistanceUS100.py

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

print('import OK!')

def detectDistance():
    distance_cm = 0
    GPIO.output(TRIG,False)
    GPIO.output(TRIG,True)
    time.sleep(0.000015)
    GPIO.output(TRIG,False)
    while GPIO.input(ECHO) == 0:
        pass
    pulseStart = time.time()
    while GPIO.input(ECHO) == 1:
        pass
    pulseEnd = time.time()
    distance_cm = (pulseEnd - pulseStart)*340/2*100
    time.sleep(1)
    return distance_cm

def averageDistance(times = 5):
    sum = 0
    dis = 0
    for i in range(times):
        sum += detectDistance()
    dis = sum/times
    return dis

try:
    GPIO.setmode(GPIO.BCM)
    TRIG = 19
    GPIO.setup(TRIG,GPIO.OUT)
    ECHO = 26
    GPIO.setup(ECHO,GPIO.IN)
    while True:
        dectDistance = detectDistance()
        print('Distance:','%.3f'% dectDistance,'cm')

except KeyboardInterrupt:
    print('Stoped by user')
    pass

finally:
    GPIO.cleanup()
    print('Clean GPIO set!')
    



