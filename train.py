# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import dirname, join, basename
import sys
from glob import glob

bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)   #对图像进行边缘检测（就是进行求导），32位浮点数，可防止溢出，1是对x求导，0是不对y求导
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)   #对图像进行边缘检测（就是进行求导），32位浮点数，可防止溢出，0是不对x求导，1是对y求导
    mag, ang = cv2.cartToPolar(gx, gy)      #计算梯度的大小和方向
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:(int)(x_pixel/2),:(int)(y_pixel/2)], bins[(int)(x_pixel/2):,:(int)(y_pixel/2)], bins[:(int)(x_pixel/2),(int)(y_pixel/2):], bins[(int)(x_pixel/2):,(int)(y_pixel/2):]
    mag_cells = mag[:(int)(x_pixel/2),:(int)(y_pixel/2)], mag[(int)(x_pixel/2):,:(int)(y_pixel/2)], mag[:(int)(x_pixel/2),(int)(y_pixel/2):], mag[(int)(x_pixel/2):,(int)(y_pixel/2):]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


img={}
num=0
for fn in glob(join(dirname(__file__)+'\cat', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    num=num+1
print ('the file path is ', dirname(__file__))
positive=num


for fn in glob(join(dirname(__file__)+'\other', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
    num=num+1

print (num,' num')
print (positive,' positive')

trainpic=[]
for i in img:
    trainpic.append(img[i])
# print(trainpic)
svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
# hogdata = [list(map(hog,img[i])) for i in img]
hogdata = list(map(hog,trainpic))
print(np.float32(hogdata).shape,' hogdata')
trainData = np.float32(hogdata).reshape(-1,bin_n*4)
print(trainData.shape,' trainData')
responses = np.float32(np.repeat(1.0,trainData.shape[0])[:,np.newaxis])
responses[positive:trainData.shape[0]]=-1.0
print(responses.shape,' responses')
print(len(trainData))
print(len(responses))
print(type(trainData))

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)


# svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
# svm.train(trainData,cv2.ml.ROW_SAMPLE,responses)
svm.save('svm_cat_data.dat')
