#!usr/bin/env python3
# !-*- coding:utf-8 -*-
# !@Time:2019/8/22 15:45
# !@Author:yuan
# !@File:getOneImageArea.py

# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

import cv2
import numpy as np

def getContours(img):
    # 中值滤波 去除椒盐噪声
    medianblur = cv2.medianBlur(img,5)
    # cv2.imshow('Median Filter Image',medianblur)
    #将RGB空间转成HSV空间，注意：OpenCV是BGR的格式，numpy是RGB的格式
    HsvImage = cv2.cvtColor(medianblur,cv2.COLOR_BGR2HSV)
    #[0 10]  [156 180]
    #设定红色的阈值
    #红色HSV：[0,43,46]-[10,255,255]
    #蓝色HSV:[100,43,46]-[124,255,255]
    rhsv_lower = np.array([0,40,56],dtype=np.uint8) #[0,70,55]
    rhsv_upper = np.array([12,255,255],dtype=np.uint8) #[10,255,255]
    #根据阈值构建掩模
    mask = cv2.inRange(HsvImage,rhsv_lower,rhsv_upper)
    cv2.imshow('After HSV Color feature filter',mask)
    #开运算,去除孤立噪声
    #卷积核(5,5)表示长宽均不足5的孤立噪声点将被去除
    #保存图片
    # cv2.imwrite('D:\\samples_withcolor\\sounrceImage_onlyColorClarifiedImage\\1120_colorclafied.png', mask)
    Kernel = np.ones((6,6),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,Kernel)
    cv2.imshow('After Opening',mask)
    #膨胀,填充孔
    Kernel1 = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask,Kernel1,iterations = 1)
    cv2.imshow('After Dilation',dilation)

    # #对原图和掩模进行按位与运算
    # res = cv2.bitwise_and(img,img,mask = mask)
    # cv2.imshow('Res',res)

    '''
    #cv2.RETR_LIST:所有轮廓，同级;
    #cv2.RETR_EXTERNAL:最外部轮廓，忽略其他轮廓;
    #cv2.RETR_CCOMP:所有轮廓，分为两级，最外一级，最外里面一级；
    #cv2.RETR_TREE:所有轮廓，分所有等级 \\
    #cv2.CHAIN_APPROX_NONE:存储轮廓所有边界点;
    #cv2.CHAIN_APPROX_SIMPLE:只存储边界点的端点 #cv2.CHAIN_APPROX_SIMPLE cv2.CHAIN_APPROX_NONE
    #cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    '''
    #cv2.findContours()会直接修改图像，所以需要复制
    imagCopy = img.copy()
    contours,hierarchy = cv2.findContours(dilation.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)


    #所有最外部轮廓进行排序
    #key = len,按列表长度进行排序，len(contours[:])
    #reverse = True,降序排序;reverse = False,升序排序
    # contours.sort(key = len,reverse = True)

    #按轮廓面积进行排序，降序排序，冒泡算法
    lenOfContours = len(contours)
    for m in range(0,lenOfContours):
        for n in range(0,lenOfContours-1-m):
            if cv2.contourArea(contours[n]) < cv2.contourArea(contours[n+1]):
                contours[n],contours[n+1] = contours[n+1],contours[n]

    # #这里可以考虑加上一个面积初值的判断，不符合即从列表中删除，实现对小面积积水不予识别
    # waterAreaMin = 2500
    # if cv2.contourArea(contours[0]) > waterAreaMin:

    contours_image = cv2.drawContours(imagCopy,[contours[0]],-1,(255,255,255),1) #(imagCopy,[contours[0]],-1,(255,255,255),1)
    #最外层轮廓面积
    area = cv2.contourArea(contours[0])
    # print(area)

    #在图像添加面积显示
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(imagCopy, 'Area:' + str(area) + 'pixels', (250, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA) #
    #显示彩色原图，并绘制带有凸缺陷的轮廓，并显示轮廓面积
    # cv2.imshow('With Defects Contours Image',imagCopy)

    #凸缺陷轮廓二值图
    withDefectsContours = np.zeros(img.shape[0:2], dtype='uint8')
    withDefectsContoursImage = cv2.drawContours(withDefectsContours, [contours[0]], -1, (255, 255, 255), 1)

    # cv2.imshow('DefectsContours Binary Image', withDefectsContoursImage)

    #轮廓的凸缺陷校正 4420.png
    # if((cv2.isContourConvex(contours[0])) == False):
    #     print('is a ao curve')
    cnt = contours[0]
    hull = cv2.convexHull(cnt,returnPoints = False)
    # print(hull)
    #返回一个数组，每一行有四个值：起点，终点，最远的点，到最远点的近似距离
    #前三个为轮廓的索引值
    defects = cv2.convexityDefects(cnt,hull)
    # print(defects)
    #修正凸缺陷轮廓，二值图
    corectDefects = np.zeros(img.shape[0:2],dtype='uint8')
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # cv2.line(imagCopy,start,end,[255,255,255],2)
        # cv2.circle(imagCopy,far,5,[0,0,255],-1)

        cv2.line(corectDefects,start,end,[255,255,255],1)
        # cv2.circle(corectDefects,far,5,[255,255,255],-1)
    # cv2.imshow('Corect Defects Binary Image',corectDefects)

    twoContoursOr = cv2.bitwise_or(withDefectsContoursImage,corectDefects)
    # cv2.imshow('Two Contours Or Operated Binary Image',twoContoursOr)

    anotherImageCopy = img.copy()  #CHAIN_APPROX_SIMPLE
    finalContours, finalHierarchy = cv2.findContours(twoContoursOr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours.sort(key = len,reverse = True)
    finalContours_image = cv2.drawContours(anotherImageCopy, [finalContours[0]], -1, (0, 0, 255), 1)
    finalArea = cv2.contourArea(finalContours[0])
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(anotherImageCopy, 'Area:' + str(finalArea) + 'pixels', (250, 460), font, 1, (0, 0, 255), 1,cv2.LINE_AA) #(390, 460)
    #凸缺陷校正后的图
    cv2.imshow('Final Contours Image', anotherImageCopy)
    #保存图片至D:\samples_withcolor\sounrceImage_onlyColorClarifiedImage
    # cv2.imwrite('D:\\samples_withcolor\\sounrceImage_onlyColorClarifiedImage\\2200_onlyColor.png',anotherImageCopy)

    #凸缺陷未校正的图
    cv2.imshow('With Defects Contours Image',imagCopy)


#D:\PyCharm\pycharm_project\textureVideoSource\afternoonWithWaterFrames\selected

#读入图像
#D:\\PyCharm\\pycharm_project\\out5frames\\20.png
#效果好:40 60  100 260 280  300 320 340 360 420  480  540 580 640 680 740 780 800 880
#效果好:1120 1180  1300 1420 1480 1580 1600 1640 1840 1880 1920  1960 2060 2080 2180 2240  2280
#       2340 2380 2480 2520 2580 2620 2680 3500 3640 3800 3920 3980 4240 4340 4380 4420 4580 4620  4680 4700
#效果差：940 1000 1080 1340(人的干扰,摄像头的饱和度的原因)
#       1240 1380 1540(摄像头不清晰） 1680 1720 1780 2000 2120 2400 2740 2780 2880(有泥垢残留)  460 2020  4380
#       识别错了:2940 2160 "2180"
#D:\\PyCharm\\pycharm_project\\out7frames\\280.png
#D:\\PyCharm\\pycharm_project\\textureVideoSource\\afternoonWithWaterFrames\\selected\\5780.png


file_path = 'D:\\PyCharm\\pycharm_project\\textureVideoSource\\originVideoAndItsFrames\\afternoonWithWaterWaterWithColor1027\\1120.png'
img = cv2.imread(file_path,cv2.IMREAD_COLOR)
cv2.imshow('Source Img',img)

# cv2.imshow('Source Img',img)
# #获取轮廓
getContours(img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

















# def getContours1(img):
#     # 中值滤波 去除椒盐噪声
#     medianblur = cv2.medianBlur(img,5)
#     # cv2.imshow('Median Filter Image',medianblur)
#     HsvImage = cv2.cvtColor(medianblur,cv2.COLOR_BGR2HSV)
#     #[0 10]  [156 180]
#     #设定红色的阈值
#     #红色HSV：[0,43,46]-[10,255,255]
#     #蓝色HSV:[100,43,46]-[124,255,255]
#     #根据阈值构建掩模
#     rhsv_lower = np.array([0,70,55],dtype=np.uint8)
#     rhsv_upper = np.array([10,255,255],dtype=np.uint8)
#     mask = cv2.inRange(HsvImage,rhsv_lower,rhsv_upper)
#     cv2.imshow('mask',mask)
#
#     #根据阈值构建掩模
#     rhsv_lower1 = np.array([150,43,46],dtype=np.uint8)
#     rhsv_upper1 = np.array([180,255,255],dtype=np.uint8)
#     mask1 = cv2.inRange(HsvImage,rhsv_lower1,rhsv_upper1)
#     cv2.imshow('Mask1', mask1)
#
#     both = cv2.bitwise_or(mask, mask1)
#     cv2.imshow('Both',both)
#
#     #开运算,去除孤立噪声
#     #卷积核(5,5)表示长宽均不足5的孤立噪声点将被去除
#     Kernel = np.ones((6,6),np.uint8)
#     both = cv2.morphologyEx(both,cv2.MORPH_OPEN,Kernel)
#     cv2.imshow('After Opening',both)
#     #膨胀,填充孔
#     Kernel1 = np.ones((5, 5), np.uint8)
#     dilation = cv2.dilate(both,Kernel1,iterations = 1)
#     cv2.imshow('After Dilation',dilation)
#     # #对原图和掩模进行按位与运算
#     # res = cv2.bitwise_and(img,img,mask = mask)
#     # cv2.imshow('Res',res)
#
#     '''
#     #cv2.RETR_LIST:所有轮廓，同级;
#     #cv2.RETR_EXTERNAL:最外部轮廓，忽略其他轮廓;
#     #cv2.RETR_CCOMP:所有轮廓，分为两级，最外一级，最外里面一级；
#     #cv2.RETR_TREE:所有轮廓，分所有等级 ||
#     #cv2.CHAIN_APPROX_NONE:存储轮廓所有边界点;
#     #cv2.CHAIN_APPROX_SIMPLE:只存储边界点的端点
#     #cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     '''
#     contours,hierarchy = cv2.findContours(both,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     contours_image = cv2.drawContours(img,contours,-1,(255,255,255),1) #img,contours,-1,(255,255,255),1
#     cv2.imshow('Contours Image',contours_image)
#     # #轮廓面积
#     # area = cv2.contourArea(contours[0])
#     # print(area)
