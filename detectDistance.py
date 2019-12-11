#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/10/14 16:37
# !@Author:yuan
# !@File:detectDistance.py

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

print('import OK!')

#单次超声测距
def detectDistance():
    #send trig for 10us,set to 15us
    GPIO.output(TRIG,True)
    time.sleep(0.000015)
    GPIO.output(TRIG,False)
    #wait low to end
    while GPIO.input(ECHO) == 0:
        pass
    #record launch time
    pulseStart = time.time()
    while GPIO.input(ECHO) == 1:
        pass
    #record return time
    pulseEnd = time.time()
    #unit:cm
    distance = (pulseEnd - pulseStart)*340/2*100
    return distance

#5次平均滤波
def averageDistance(times = 5):
    sum = 0
    for i in range(times):
        sum += detectDistance()
        #测量间隔建议60ms = 0.06s以上
        time.sleep(0.06)
    return (sum/times)

'''
#限幅平均滤波
downLimit:测量值的下限(cm)
upLimit:测量值的上限(cm)
error:相邻测量值的偏差最大值(cm)
countOutTimes:# 最大超时次数,默认为2
返回值:5次限幅平均滤波的平均值
'''
def limitAverageDistance(downLimit,upLimit,error,countOutTimes = 2):
    countOutTimes = 2
    count = 0  # 次数计数
    limAvrDistance = 0

    a1 = detectDistance()
    #测量间隔建议60ms = 0.06s以上
    time.sleep(0.06)
    while a1 < downLimit or a1 > upLimit:
        a1 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            break
    count = 0

    a2 = detectDistance()
    time.sleep(0.06)
    while a2 < downLimit or a2 > upLimit:
        a2 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            break
    count = 0
    while abs(a2-a1) > error:
        a2 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            a2 = a1
            break
    count = 0

    a3 = detectDistance()
    time.sleep(0.06)
    while a3 < downLimit or a3 > upLimit:
        a3 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            break
    count = 0
    while abs(a3-a2) > error:
        a3 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            a3 = a2
            break
    count = 0

    a4 = detectDistance()
    time.sleep(0.06)
    while a4 < downLimit or a4 > upLimit:
        a4 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            break
    count = 0
    while abs(a4-a3) > error:
        a4 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            a4 = a3
            break
    count = 0

    a5 = detectDistance()
    time.sleep(0.06)
    while a5 < downLimit or a5 > upLimit:
        a5 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            break
    count = 0
    while abs(a5-a4) > error:
        a5 = detectDistance()
        time.sleep(0.06)
        count += 1
        if count == countOutTimes:
            a5 = a4
            break
    count = 0

    limAvrDistance = (a1+a2+a3+a4+a5)/5
    return limAvrDistance



# if __name__ == '__main__':
try:
    '''
    #GPIO初始化
    '''
    #set GPIO mode to BCM
    GPIO.setmode(GPIO.BCM)
    #set GPIO pin
    TRIG = 19
    ECHO = 26
    #set GPIO mode
    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)

    dis = averageDistance()
    print('averageDistance:','%.3f'% dis,'cm')

    dis1 = limitAverageDistance(20,100,50)
    print('limitAverageDistance:','%.3f'% dis1,'cm')

    #reset by Ctrl+c
except KeyboardInterrupt:
    print('stoped by user')
    pass

finally:
    GPIO.cleanup()
    print('clean GPIO set!')

        
