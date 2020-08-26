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

def fieldInsideField(f1,f2):
    if(f1[0]>=f2[0] and f1[0]<=f2[2] and f1[1]>=f2[1] and f1[1]<=f2[3]):
        return True
    return False

def isThereSameRect(rect,contures):
    xR, yR, wR, hR = cv2.boundingRect(rect)
    for i in range(len(contures)):
        if(contures[i] is not None):
            c=contures[i]
            x, y, w, h = cv2.boundingRect(c)
            if(xR!=x or yR!=y or wR!=w or hR!=h):
                xMis=abs(xR-x)
                yMis=abs(yR-y)
                wMis= abs(wR-w)
                hMis= abs(hR-h)
                misRange=20
                if(xMis<=misRange and yMis<=misRange):
                    contures[i]=None
                    return False
                elif(xMis<=misRange and yMis<=misRange and wMis<=misRange and hMis<=misRange):
                    return True
    return False

def retPotentialsFields(fileName):
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
                if(x<5and y<5):
                    contours[i] = None
                if(w>60 and h>120):
                    # approximate the contour
                    peri = cv2.arcLength(hull, True)
                    errorSign=0.01
                    for i in range(3):
                        approx = cv2.approxPolyDP(hull, errorSign* peri, True)
                        if len(approx)>=4 or len(approx)<=6:
                            #rect = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            rects.append([x,y,x+w,y+h])
                            break
                        errorSign+=0.01
            else:
                contours[i]=None

    for i in rects:
        rect = cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 4)
    for i in range(len(rects)):
        for k in range(len(rects)):
            if(rects[i]!=1 and rects[k]!=1 and rects[i]!=rects[k] and fieldInsideField(rects[i],rects[k])):
                rects[k]=1
    while(rects.count(1)>0):
        rects.remove(1)
    for i in rects:
        rect = cv2.rectangle(im, (i[0],i[1]), (i[2], i[3]), (0, 0, 255), 3)
    #cv2.imshow('image', im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return rects

#filename = 'pics\\screenshotedImages\\word1.png'
#word1
#portal
#DONEpitCreateTicket2
#DONE2018-08-31 (18)
#Cisco Finness
#im = cv2.imread(filename)
#potentialRects=retPotentialsFields(im)

#print(potentialRects)
