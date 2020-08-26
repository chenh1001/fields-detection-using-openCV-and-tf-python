import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import pickle
#im = cv2.imread('login_form.png')
#im = cv2.imread('login2.png')
#im = cv2.imread('login3.jpeg')
#im=cv2.imread('login4.png')
#im=cv2.imread('login5.png')
#im=cv2.imread('login6.jpg')
#im=cv2.imread('login7.jpg')
#im=cv2.imread('login8.png')
#im=cv2.imread('login9.jpg')
#im=cv2.imread('login10.png')
#im=cv2.imread('login11.png')
file={1 :'login2.png',2:'login3.jpeg',3:'login4.png',4:'login5.png',5:'login6.jpg',6:'login7.jpg',7:'login8.png',8:'login9.jpg',9:'login10.png',10:'login11.png',11:'login_form.png'}
strNum=random.randrange(1,len(file)+1)
#im=cv2.imread(file[strNum])
#im=cv2.imread('E:\screenshotedImages\\2018-08-31 (46).png')

def isThereSameRect(rect,contures):
    xR, yR, wR, hR = cv2.boundingRect(rect)
    for i in range(len(contures)):
        if(contures[i] is not None):
            c=contures[i]
            x, y, w, h = cv2.boundingRect(c)
            if(xR!=x and yR!=y and wR!=w and hR!=h):
                xMis=abs(xR-x)
                yMis=abs(yR-y)
                wMis= abs(wR-w)
                hMis= abs(hR-h)
                if(xMis<=5 and yMis<=5):
                    contures[i]=None
                    return False
                elif(xMis<=5 and yMis<=5 and wMis<=5 and hMis<=5):
                    return True
    return False

def retPotentials(fileName):
    #im = cv2.imread(fileName)
    im=fileName
    #im= im[90:-80,:]
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow('imgB4',th2)
    im2, contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for i in range(len(contours)):
        c=contours[i]
        if(contours[i] is not None):
            if(isThereSameRect(c,contours)==False):
                hull = cv2.convexHull(c)
                x, y, w, h = cv2.boundingRect(hull)
                if(w>20 and h>10):
                    # approximate the contour
                    peri = cv2.arcLength(hull, True)
                    errorSign=0.01
                    for i in range(3):
                        approx = cv2.approxPolyDP(hull, errorSign* peri, True)
                        if len(approx)>=4 or len(approx)<=6:
                            #rect = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            rects.append([x,y,x+w,y+h])
                            break
                        errorSign+=0.01
            else:
                contours[i]=None


    #cv2.imshow('image', im)
    #cv2.waitKey(0)
    return rects

#potentialRects=retPotentials('E:\screenshotedImages\\DONE2018-08-31 (8).png')

#print(potentialRects)
