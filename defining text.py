import numpy as np
from datetime import datetime
import cv2
import time
import pathlib
import pickle
import math
#import pytesseract
import os
from PIL import Image

global realImg
global img
global mode
global startingX1,startingY1,startingX2,startingY2,endingX1,endingY1,endingX2,endingY2,isSquare1,isSquare2
startingX1,startingY1,startingX2,startingY2,endingX1,endingY1,endingX2,endingY2,isSquare1,isSquare2=0,0,0,0,0,0,0,0,0,0
drawing = False
ix,iy = -1,-1
mode=0
global definingTrueMatches
definingTrueMatches=3    #if defining false matches change to 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 0  TEXT  |=====|
# 1 TEXT   TEXT
# 2 TEXT
#   |=====|
# 3 TEXT
#   TEXT
#pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Admin\\AppData\\Local\\Tesseract-OCR\\tesseract"
#pytesseract.pytesseract.tesseract_cmd="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"

def angleFromPoints(x1,y1,x2,y2):
    angle=0
    if x1==x2:
        angle=0
    else:
        angle=(y2-y1)/(x2-x1)
    angle=math.atan2((y2-y1),(x2-x1))
    #print(math.degrees(angle))
    return angle


def isRectInImage(filename):
    im = cv2.imread(filename)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    im2, contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if(i!=0):
            c=contours[i]
            hull = cv2.convexHull(c)
            x, y, w, h = cv2.boundingRect(hull)
            if (w > 10 and h > 5):
                # approximate the contour
                peri = cv2.arcLength(hull, True)
                errorSign = 0.01
                for i in range(3):
                    approx = cv2.approxPolyDP(hull, errorSign * peri, True)
                    if len(approx) >= 4 and len(approx) <= 5:
                        return True
                    errorSign += 0.01
            return False

def addItemToList():
    global startingX1, startingY1, startingX2, startingY2, endingX1,endingY1,endingX2,endingY2,definingTrueMatches,isSquare1,isSquare2
    angleStarting=angleFromPoints(startingX1,startingY1,startingX2,startingY2)
    angleEnding=angleFromPoints(endingX1,endingY1,endingX2,endingY2)
    #distStarting=math.sqrt((startingX1-startingX2)*(startingX1-startingX2)+(startingY1-startingY2)*(startingY1-startingY2))
    #distEnding=math.sqrt((endingX1-endingX2)*(endingX1-endingX2)+(endingY1-endingY2)*(endingY1-endingY2))
    xStarting=startingX2-startingX1
    yStarting=startingY2-startingY1
    xEnding=endingX2-endingX1
    yEnding=endingY2-endingY1
    hight1=endingY1-startingY1
    hight2=endingY2-startingY2
    if(definingTrueMatches==0):
        isSquare1=0
        isSquare2=1
    elif (definingTrueMatches == 1):
        isSquare1 = 0
        isSquare2 = 0
    elif (definingTrueMatches == 2):
        isSquare1 = 0
        isSquare2 = 1
    elif (definingTrueMatches == 3):
        isSquare1 = 0
        isSquare2 = 0
    inputToFinalNeuralNetwork.append([angleStarting,angleEnding,xStarting/(endingX1-startingX1),yStarting/hight1,xEnding/(endingX2-startingX2),yEnding/hight2,hight1,hight2,isSquare1,isSquare2,definingTrueMatches])
    f = open('WorkData.pckl', 'wb')
    pickle.dump(inputToFinalNeuralNetwork, f)
    f.close()
    k= open('WorkDataBackUp.pckl','wb')
    backUpPoints.append([startingX1, startingY1, endingX1, endingY1,startingX2, startingY2, endingX2, endingY2,isSquare1,isSquare2,definingTrueMatches]) #  add is square or not
    pickle.dump(backUpPoints,k)
    k.close()
    arrMone[definingTrueMatches]+=1
    n = open('Mone.pckl', 'wb')
    pickle.dump(arrMone, n)
    n.close()
    startingX1, startingY1, startingX2, startingY2, endingX1, endingY1, endingX2, endingY2 = 0, 0, 0, 0, 0, 0, 0, 0
    print(inputToFinalNeuralNetwork[-1],backUpPoints[-1])
    print(isSquare1,isSquare2)

def draw_rect(event,x,y,flags,param):
    global ix,iy,drawing,img,mode,startingX1, startingY1, startingX2, startingY2, endingX1,endingY1,endingX2,endingY2,isSquare1,isSquare2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), 1)
            cv2.imshow('image', img)
            img=cv2.imread('img.png')

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.imshow('image', realImg)
        if(y>iy):
            a=y
            b=iy
        else:
            a=iy
            b=y
        if (x > ix):
            c = x
            d = ix
        else:
            c = ix
            d = x
        img=img[b:a,d:c]
        if(mode==0):
            timestamp=str(datetime.now().time().hour+datetime.now().time().minute+datetime.now().time().microsecond)
            fileplace=str("pics\\defenitions\\"+timestamp+'.png')
            cv2.imwrite(fileplace, img)
            cv2.rectangle(img, (0, 0), (512, 512), (255, 255, 255), -1)
            #if isRectInImage(fileplace)==True:
            #    isSquare1=1
            #else:
            #    isSquare1=0
            startingX1=d
            startingY1=b
            endingX1=c
            endingY1=a
        elif(mode==1):
            timestamp=str(datetime.now().time().hour+datetime.now().time().minute+datetime.now().time().microsecond)
            fileplace=str("pics\\Squares\\"+timestamp+'.png')
            cv2.imwrite(fileplace, img)
            cv2.rectangle(img, (0, 0), (512, 512), (255, 255, 255), -1)

            #if isRectInImage(fileplace)==True:
            #    isSquare2=1
            #else:
            #    isSquare2=0
            startingX2 = d
            startingY2 = b
            endingX2 = c
            endingY2 = a
        img= cv2.imread('img.png')
        mode+=1
        if(mode>=2):
            addItemToList()
            mode=0

inputToFinalNeuralNetwork=[]
#f=open('WorkData.pckl','wb')
#pickle.dump(inputToFinalNeuralNetwork,f)
#f=open('WorkDataBackUp.pckl','wb')
#pickle.dump(inputToFinalNeuralNetwork,f)
# f=open('Mone.pckl','wb')
#pickle.dump(arrMone,f)

arrMone =[]
n= open('Mone.pckl','rb')
arrMone=pickle.load(n)
n.close()
print(arrMone)

f= open('WorkData.pckl','rb')
inputToFinalNeuralNetwork=pickle.load(f)
f.close()
k= open('WorkDataBackUp.pckl','rb')
backUpPoints=pickle.load(k)
k.close()
print(len(backUpPoints))
print(backUpPoints)
print(len(inputToFinalNeuralNetwork))
print(inputToFinalNeuralNetwork)
#newList=np.array(inputToFinalNeuralNetwork)
#print(newList/10)

directOfFullImages=pathlib.Path('pics\\screenshotedImages')
'''     
for fileName in os.listdir('pics\\screenshotedImages'):
    first4 = fileName[0:4]
    if(first4=='DONE'):
        unDone = fileName[4:]
        os.renames(str('pics\\screenshotedImages\\' + fileName), str('pics\\screenshotedImages\\' + unDone))
'''
for fileName in os.listdir('pics\\screenshotedImages'):
    first4 = fileName[0:4]
    if(len(fileName)<4 or (first4!='DONE' and first4!='DANA')):
        print(fileName,first4)
        img=cv2.imread(str('pics\\screenshotedImages\\'+fileName))
        img = img[90:-80, :]
        realImg=cv2.imread(str('pics\\screenshotedImages\\'+fileName))
        realImg=realImg[90:-80,:]
        cv2.imwrite('img.png',realImg)
        cv2.namedWindow('image')
        cv2.imshow('image',img)
        cv2.setMouseCallback('image', draw_rect)
        finished=False
        while(finished==False):
            k=chr(cv2.waitKey(0))
            if k == ' ' or k=='':
                cv2.destroyAllWindows()
                finished=True
                mode=0
            elif k=='z':
                k = open('WorkData.pckl', 'rb')
                inputToFinalNeuralNetwork = pickle.load(k)
                k.close()
                inputToFinalNeuralNetwork.pop()
                f = open('WorkData.pckl', 'wb')
                pickle.dump(inputToFinalNeuralNetwork, f)
                f.close()
                print(inputToFinalNeuralNetwork)

                points=[]
                n = open('WorkDataBackUp.pckl', 'rb')
                points = pickle.load(n)
                n.close()
                points.pop()
                c = open('WorkDataBackUp.pckl', 'wb')
                pickle.dump(points, c)
                c.close()
                print(points)
                arrMone[definingTrueMatches] -= 1

        os.renames(str('pics\\screenshotedImages\\'+fileName),str('pics\\screenshotedImages\\DANA'+fileName))
        ix=-1
        iy=-1
    else:
        if (len(fileName) > 4  and first4 == 'DANA'):
            unDone=fileName[4:]
            os.renames(str('pics\\screenshotedImages\\' + fileName), str('pics\\screenshotedImages\\' + unDone))


'''while(1): 
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('c'):
        cv2.destroyAllWindows()
        time.sleep(1)
        im = PIL.ImageGrab.grab()
        im.save('screenPic.png')
        retPotentials('screenPic.png')
        cv2.namedWindow('image')
        cv2.rectangle(img, (0, 0), (400, 400), (255, 255, 255), -1)
        cv2.imshow('image', img)
        print('got it')'''