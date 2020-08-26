import numpy as np
import pickle
import cv2
import tflearn
import math
import pytesseract
import os
from PIL import Image
from random import shuffle
from definingSquares import retPotentials
from testTextOCR import testOCR
from datetime import datetime
'''
Squere:
Hight
Width
Text:
Hight
Width
abs(X1-x2)
abs(Y1-y2)
isCoupleOrNot (0,1)
    '''
def isRectInList(rect,list):
    for a in list:
        if rect==a:
            return True
    return False
def areInRightOrder(mainRect,otherRect):
    if mainRect[0]-otherRect[0]<mainRect[2]-mainRect[0] or mainRect[1]-otherRect[1]<mainRect[3]-mainRect[1]:
        return True
    return False
def angleFromPoints(x1,y1,x2,y2):
    angle=0
    if x1==x2:
        angle=0
    else:
        angle=(y2-y1)/(x2-x1)
    angle=math.atan2((y2-y1),(x2-x1))
    #print(math.degrees(angle))
    return angle
def isThereDotsInImage(filename):
    arrOfChars=pytesseract.image_to_string(Image.open(filename))
    for i in arrOfChars[-5:]:
        if i==':':
            return True
    return False

def areTwoInputsClose(in1,in2):
    numOfSimilar=0
    for i in range(len(in1)):
        if i!=2 and i!=3 and abs(in1[i]-in2[i])<=0.1*in1[i]:
            numOfSimilar+=1
    if numOfSimilar>=3:
        return True
    else:
        return False

#pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\{}\\AppData\\Local\\Tesseract-OCR\\tesseract".format(os.getlogin())
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"

k= open('backUpData.pckl','rb')
data=pickle.load(k)
k.close()
def getInputFromTwoRects(r1,r2):
    startingX1 = r1[0]
    startingY1 = r1[1]
    endingX1 = r1[2]
    endingY1 = r1[3]
    startingX2 = r2[0]
    startingY2 = r2[1]
    endingX2 = r2[2]
    endingY2 = r2[3]
    angleStarting = angleFromPoints(startingX1, startingY1, startingX2, startingY2)
    angleEnding = angleFromPoints(endingX1, endingY1, endingX2, endingY2)
    distStarting = math.sqrt(
        (startingX1 - startingX2) * (startingX1 - startingX2) + (startingY1 - startingY2) * (startingY1 - startingY2))
    distEnding = math.sqrt(
        (endingX1 - endingX2) * (endingX1 - endingX2) + (endingY1 - endingY2) * (endingY1 - endingY2))

    isSquare1 = isRectInList(rects[k], rectsFromSquares)
    isSquare2 = isRectInList(rects[n], rectsFromSquares)

    pred = [angleStarting, angleEnding, distStarting / abs(endingX1 - startingX1),distEnding / abs(endingX2 - startingX2), isSquare1, isSquare2]
    return pred

for i in data:
    i.pop(4)
    i.pop(4)
#print(data)
#for i in range(len(data)): #PROCESS DATA
#    xb= data[i][5]-data[i][1]+data[i][3]
#    yb= data[i][4]-data[i][0]+data[i][2]
#    data[i]=[abs(data[i][5]),abs(data[i][4]),abs(xb),abs(yb),data[i][6]]
#print(data)

#print(lastOnes)
#for i in lastOnes:
#    data.append(i)
#for i in lastOnes:
#    data.append(i)
#for i in lastOnes:
#    data.append(i)
#data=data[:-200]
shuffle(data)
#print(data)

labels=[]
for i in (data):
    if(i[6]==0):
        labels.append([1,0,0,0])
    elif (i[6]==1):
        labels.append([0,1,0,0])
    elif (i[6]==2):
        labels.append([0,0,1,0])
    elif (i[6]==3):
        labels.append([0,0,0,1])
    else:
        print('wtfff')

#print(data)
#print(labels)
for i in data:
    if(len(i)==7):
        i.remove(i[6])

filename='pics\\screenshotedImages\\AppOpsPortal_DivDNSend.png'
#filename='testNN.png'
#DONE2018-08-31 (32)
#DONEAppOpsPortal_DivRecipientAddress
# DONE2018-08-31 (8)
im=cv2.imread(filename)
im= im[90:-80,:]
rects=[]
rectsFromSquares = retPotentials(im)
rectsFromOCR = testOCR(im)
for a in rectsFromSquares:
    rects.append(a)
for a in rectsFromOCR:
    rects.append(a)

#print(rects)

for rect in rects:
    cv2.rectangle(im,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),1)
'''listOfSimilarCouples=[]
for k in range(len(rects)):
    for n in range(len(rects)):
        if k==0 and n==0:
            listOfSimilarCouples.append([])
            listOfSimilarCouples[0].append([k,n,getInputFromTwoRects(rects[k],rects[n])])
        elif k!=n:
            input=getInputFromTwoRects(rects[k],rects[n])
            foundCouple=False
            for i in range(len(listOfSimilarCouples)):
                if areTwoInputsClose(input,listOfSimilarCouples[i][0][2]):
                    listOfSimilarCouples[i].append([k,n,input])
                    foundCouple=True
                    break
            if foundCouple==False:
                listOfSimilarCouples.append([])
                listOfSimilarCouples[-1].append([k,n,input])
maxLen=0
maxList=[]
for i in listOfSimilarCouples:
    if len(i)>maxLen:
        maxLen=len(i)
        maxList=i
print(listOfSimilarCouples)
print(len(listOfSimilarCouples))
print(maxList)'''
net = tflearn.input_data(shape=[None,6])
net = tflearn.fully_connected(net, 500,activation='ReLU')
net = tflearn.fully_connected(net, 500,activation='ReLU')
net = tflearn.fully_connected(net, 500,activation='ReLU')
net = tflearn.fully_connected(net, 500,activation='ReLU')
net = tflearn.fully_connected(net, 4, activation='Sigmoid')
#net = tflearn.fully_connected(net, 4, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
#model.fit(data, labels, n_epoch=200,batch_size=20,show_metric=True,shuffle=True)
model.load("newDataM2")
#model.load("model.tfl")
#model.load("modelV2.tfl")
#model.load("testingNewData.tfl")
goodMatches=[]
for k in range(len(rects)):
    if(rects[k]!=None):
        bestMatch=[]
        bestPrediction=0
        bestIndex=0
        isRectFromRects = isRectInList(rects[k],rectsFromSquares)
        bestMatchIndex=0
        if isRectFromRects==False:
            for n in range(len(rects)):
                if(rects[n]!=None):
                    if rects[n]!=rects[k] and not((abs(rects[k][0] - rects[n][0]) < 5 and abs(rects[k][1] - rects[n][1]) < 5)):
                        pred=getInputFromTwoRects(rects[k],rects[n])
                        prediction= model.predict([pred])
                        maxPred=0
                        maxIndex=0
                        #for i in range(len(prediction[0])):
                        priorotizedIndex=3
                        if prediction[0][priorotizedIndex]>=maxPred:
                            maxPred=prediction[0][priorotizedIndex]
                            maxIndex=priorotizedIndex
                        if maxPred>bestPrediction:
                            bestPrediction=maxPred
                            bestMatch = rects[n]
                            bestMatchIndex=n
                            bestIndex=maxIndex
            if bestPrediction>=0.9:
                print(bestPrediction,bestIndex)
                goodMatches.append([rects[k],bestMatch])
                cv2.circle(im,(rects[k][0],rects[k][1]),5,(0,0,0))
                cv2.line(im, (rects[k][0], rects[k][1]), (bestMatch[0], bestMatch[1]), (0, 0, 255), 2)
                #print(rects[k], bestMatch, "rect and best match")
                rects[k]=None
                rects[bestMatchIndex]=None

        #else:
        #    cv2.rectangle(im,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,0),3)

cv2.imshow('image',im)
cv2.waitKey(0)

#ADD IS SQUARE OR NOT

