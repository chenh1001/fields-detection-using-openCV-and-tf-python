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
from findingFields import retPotentialsFields
from datetime import datetime
import random
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


def isRectInList(rect, list):
    for a in list:
        if rect == a:
            return True
    return False


def areInRightOrder(mainRect, otherRect):
    if mainRect[0] - otherRect[0] < mainRect[2] - mainRect[0] or mainRect[1] - otherRect[1] < mainRect[3] - mainRect[1]:
        return True
    return False


def angleFromPoints(x1, y1, x2, y2):
    angle = 0
    if x1 == x2:
        angle = 0
    else:
        angle = (y2 - y1) / (x2 - x1)
    angle = math.atan2((y2 - y1), (x2 - x1))
    # print(math.degrees(angle))
    return angle


def isThereDotsInImage(filename):
    arrOfChars = pytesseract.image_to_string(Image.open(filename))
    for i in arrOfChars[-5:]:
        if i == ':':
            return True
    return False

'''
def getCurrentBestCouple():
    bestPrediction = 0
    bestMatch = []

    for k in range(len(rects)):
        if (rects[k] != None):
            for n in range(len(rects)):
                if (rects[n] != None):
                    if rects[n] != rects[k] and not((abs(rects[k][0] - rects[n][0]) < 5 and abs(rects[k][1] - rects[n][1]) < 5)):
                        pred = getInputFromTwoRects(rects[k], rects[n])
                        prediction = model.predict([pred])
                        maxPred = 0
                        # for i in range(len(prediction[0])):
                        priorotizedIndex = 3
                        if prediction[0][priorotizedIndex] >= maxPred:
                            maxPred = prediction[0][priorotizedIndex]
                        if maxPred > bestPrediction:
                            bestPrediction = maxPred
                            bestMatch = [k, n,bestPrediction]
    return bestMatch'''


pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\{}\\AppData\\Local\\Tesseract-OCR\\tesseract".format(os.getlogin())
#pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"

k = open('finalDataCombined.pckl', 'rb')
data = pickle.load(k)
k.close()

def getPredFromRects(r1,r2,priorotizedIndex,rectsFromSquares):
    pred = getInputFromTwoRects(r1, r2,rectsFromSquares)
    prediction = model.predict([pred])
    return prediction[0][priorotizedIndex]

def getInputFromTwoRects(r1, r2,rectsFromSquares):
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

    xStarting = startingX2 - startingX1
    yStarting = startingY2 - startingY1
    xEnding = endingX2 - endingX1
    yEnding = endingY2 - endingY1
    hight1 = endingY1 - startingY1
    hight2 = endingY2 - startingY2

    isSquare1 = isRectInList(r1, rectsFromSquares)
    isSquare2 = isRectInList(r2, rectsFromSquares)
    if isSquare1==True:
        isSquare1=1
    else:
        isSquare1=0
    if isSquare2==True:
        isSquare2=1
    else:
        isSquare2=0
        #if |||
        #   ||||||
        # 0 2 3 5 6 7 8 9
        #if ||| ||||||||
        # 0 1 3 5 6 7 8 9
    pred = [angleStarting, angleEnding, xStarting / (endingX1 - startingX1), yStarting / hight1,xEnding / (endingX2 - startingX2), yEnding / hight2, hight1, hight2, isSquare1, isSquare2]
    return pred
def removeAllpredictionsWithThisRects(k,n,arr):
    for i in range(len(arr)):
        if arr[i]!=None:
            if arr[i][0] ==k or arr[i][0]==n or arr[i][1]==k or arr[i][1]==n:
                arr[i]=None

def areTwoInputsClose(in1,in2,pIndex):
    numOfSimilar=0
    if(pIndex==0):
        for i in range(len(in1)):
            if (i!=3 and i != 4 and i!=5):
                if in1[i]==in2[i] or (abs(in1[i]-in2[i])<=0.5*(abs(in1[i]+in2[i])/2)) or (abs(in1[i])<0.3 and abs(in2[i])<0.3):
                    numOfSimilar+=1
        if numOfSimilar>=7:
            return True
        else:
            return False
    elif(pIndex==1):
        for i in range(len(in1)):
            if (i != 2 and i != 4 and i!=5):
                if in1[i]==in2[i] or (abs(in1[i]-in2[i])<=0.4*(abs(in1[i]+in2[i])/2)) or (abs(in1[i])<0.3 and abs(in2[i])<0.3):
                    numOfSimilar+=1
        if numOfSimilar>=7:
            return True
        else:
            return False
    elif(pIndex==2):
        for i in range(len(in1)):
            if(i!=1 and i!=4 and i!=5):
                if in1[i]==in2[i] or (abs(in1[i]-in2[i])<=0.4*(abs(in1[i]+in2[i])/2)) or (abs(in1[i])<0.3 and abs(in2[i])<0.3):
                    numOfSimilar+=1
        if numOfSimilar>=7:
            return True
        else:
            return False
    elif (pIndex == 3):
        for i in range(len(in1)):
            if (i != 1 and i != 4):
                if in1[i] == in2[i] or (abs(in1[i] - in2[i]) <= 0.3 * (abs(in1[i] + in2[i]) / 2)) or (abs(in1[i]) < 0.3 and abs(in2[i]) < 0.3):
                    numOfSimilar += 1
                elif((i==6 or i==7) and abs(in1[i]-in2[i])<0.4*(abs(in2[i]))):
                    numOfSimilar+=1
                elif (i == 3 and (abs(in1[5] - in2[5]) <= 0.2 * (abs(in1[5] + in2[5]) / 2))):
                    numOfSimilar += 1
                elif(i==5 and not (abs(in1[3] - in2[3]) <= 0.2 * (abs(in1[3] + in2[3]) / 2))):
                    numOfSimilar+=1

        if numOfSimilar >= 8:
            return True
        else:
            return False
#[1, 2, [1.633215136790854, 1.5083775167989393, -0.020833333333333332, 2.0, 0.02, 2.0, 8, 8, 0, 0]
#[3, 4, [1.633215136790854, 0.6000502134017536, -0.00847457627118644, 1.4545454545454546, 0.13768115942028986, 1.625, 11, 8, 0, 0]
#[5, 6, [1.633215136790854, 2.214297435588181, -0.018867924528301886, 2.0, -0.2857142857142857, 2.0, 8, 8, 0, 0], 0.993103]
#[9, 10, [1.5707963267948966, 0.17573829668536678, 0.0, 2.0, 0.6149425287356322, 1.7272727272727273, 8, 11, 0, 0], 0.9932679]
#for i in data:
#    i.pop(4)
#    i.pop(4)
print(data)
# for i in range(len(data)): #PROCESS DATA
#    xb= data[i][5]-data[i][1]+data[i][3]
#    yb= data[i][4]-data[i][0]+data[i][2]
#    data[i]=[abs(data[i][5]),abs(data[i][4]),abs(xb),abs(yb),data[i][6]]
# print(data)

# print(lastOnes)
# for i in lastOnes:
#    data.append(i)
# for i in lastOnes:
#    data.append(i)
# for i in lastOnes:
#    data.append(i)
# data=data[:-200]
shuffle(data)
# print(data)

labels = []
counter=0
print(len(data))
for i in (data):
    if (i[10] == 0):
        labels.append([1, 0, 0, 0])
    elif (i[10] == 1):
        labels.append([0, 1, 0, 0])
    elif (i[10] == 2):
        labels.append([0, 0, 1, 0])
    elif (i[10] == 3):
        labels.append([0, 0, 0, 1])
    else:
        print('wtfff')

print(data)
print(labels)
for i in data:
    if (len(i) == 11):
        i.pop()
'''test=[]
testLabels=[]
for i in range(50):
    test.append(data[-1])
    data.pop()
    testLabels.append(labels[-1])
    labels.pop()'''
#GR Web
net = tflearn.input_data(shape=[None, 10])
net = tflearn.fully_connected(net, 100, activation='ReLU')
net = tflearn.fully_connected(net, 100, activation='ReLU')
net = tflearn.fully_connected(net, 40, activation='ReLU')
net = tflearn.fully_connected(net, 10, activation='ReLU')
net = tflearn.fully_connected(net, 4, activation='sigmoid')
net = tflearn.regression(net)
#net = tflearn.fully_connected(net, 4, activation='softmax')

model = tflearn.DNN(net)
#model.fit(data, labels, n_epoch=50,batch_size=20,show_metric=True,shuffle=True)#,validation_set=0.1)
#model.save("binar.tfl")
model.load("bettaModelSigmoid.tfl")

def getCouples(fileName,priorotizedIndex=1):
    # filename='testNN.png'
    # DONE2018-08-31 (32)
    # AppOpsPortal_DivDNSend
    # DONE2018-08-31 (8)
    #im = cv2.imread(fileName)
    im=fileName
    #im = im[90:-80, :]
    rects = []
    rectsFromSquares = retPotentials(im)
    rectsFromOCR = testOCR(im)
    for a in rectsFromSquares:
        rects.append(a)
    for a in rectsFromOCR:
        rects.append(a)

    # print(rects)

    #for rect in rects:
    #    cv2.rectangle(im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)

    #model.load("newDataM2")
    # model.load("model.tfl")
    # model.load("modelV2.tfl")
    # model.load("testingNewData.tfl")
    #model.load("modelNewData2.tfl")

    rectsPredictions=[]
    for k in range(len(rects)):
        isRectFromRects = isRectInList(rects[k], rectsFromSquares)
        if isRectFromRects == False:
            for n in range(len(rects)):
                if rects[n] != rects[k]:
                    rectsPredictions.append([k,n,getPredFromRects(rects[k],rects[n],priorotizedIndex,rectsFromSquares)])

    threshhold=0.1
    prediction=1
    rectsPredictions.sort(key=lambda x:x[2])
    potCouples=[x for x in rectsPredictions if x[2] >= threshhold]

    print(rectsPredictions)
    ################################################################################################################################

    '''for p in range(a): #(len(potCouples)):
        k = potCouples[p][0]
        n = potCouples[p][1]
        cv2.circle(im, (rects[k][0], rects[k][1]), 5, (0, 0, 0))
        cv2.line(im, (rects[k][0], rects[k][1]), (rects[n][0], rects[n][1]), (0, 0, 0), 1)'''

    listOfSimilarCouples=[]
    for p in range(len(potCouples)):
        k=potCouples[p][0]
        n=potCouples[p][1]
        pred=potCouples[p][2]
        if p == 0:
            listOfSimilarCouples.append([])
            listOfSimilarCouples[0].append([k,n, getInputFromTwoRects(rects[k], rects[n],rectsFromSquares),pred])
        else:
            input = getInputFromTwoRects(rects[k], rects[n],rectsFromSquares)
            foundCouple = False
            for i in range(len(listOfSimilarCouples)):
                if areTwoInputsClose(input, listOfSimilarCouples[i][0][2], priorotizedIndex):
                    listOfSimilarCouples[i].append([k, n, input, pred])
                    foundCouple = True
                    break

            if foundCouple == False:
                listOfSimilarCouples.append([])
                listOfSimilarCouples[-1].append([k, n, input,pred])
    maxLen=0
    maxPred=0
    maxList=[]
    listAfterNipoi=[]
    listAfterNipoi.append([])

    for i in listOfSimilarCouples:
        i.reverse()
    for i in listOfSimilarCouples:
        for p in i:
            if(p!=None):
                listAfterNipoi[-1].append(p)
                k = p[0]
                n = p[1]
                removeAllpredictionsWithThisRects(k, n, i)
        listAfterNipoi.append([])

    for i in listAfterNipoi:
        if len(i)>maxLen and i[-1][3]+0.1>=maxPred:
            maxLen=len(i)
            maxList=i
            maxPred=i[-1][3]
    #print(listAfterNipoi)
    #for i in listAfterNipoi:
    #    print(len(i))
    print(maxList)
    print(len(maxList))
    maxList.reverse()
    retCouples=[]
    while prediction>=threshhold:
        if len(maxList)>0:
            if maxList[-1]!=None:
                #k,n,prediction=maxList[-1]
                k=maxList[-1][0]
                n=maxList[-1][1]
                prediction=maxList[-1][3]
                if prediction>=threshhold:
                    retCouples.append([rects[k],rects[n]])
                    maxList.pop()
                    removeAllpredictionsWithThisRects(k,n,maxList)
            else:
                maxList.pop()
        else:
            break
    return retCouples

def getAllCoupelesOnImage(filename):
    fields=retPotentialsFields(filename)
    rects=[]
    for field in fields:
        fieldX=field[0]
        fieldY=field[1]
        img = filename[field[1]:field[3], field[0]:field[2]]#y1:y2,x1:x2
        #timestamp = str(datetime.now().time().hour + datetime.now().time().minute + datetime.now().time().microsecond)
        name="temp"
        fileplace = str("pics\\defenitions\\" + name + '.png')
        cv2.imwrite(fileplace, img)
        imOfField=cv2.imread(fileplace)
        rectsToAdd=getCouples(imOfField,2) ## PRIOTOTIZED INDEX
        for i in rectsToAdd: #adjust to window
            i[0][0] += fieldX
            i[0][1] += fieldY
            i[0][2] += fieldX
            i[0][3] += fieldY
            i[1][0] += fieldX
            i[1][1] += fieldY
            i[1][2] += fieldX
            i[1][3] += fieldY
        rects+=rectsToAdd
    return rects

'''
#GR Web
filename = 'pics\\screenshotedImages\\GR Web.png'
im = cv2.imread(filename)
rects=getCouples(filename,2)
for i in rects:
    cv2.rectangle(im, (i[0][0], i[0][1]), (i[0][2], i[0][3]),(0,255,0), 1)
    cv2.rectangle(im, (i[1][0], i[1][1]), (i[1][2], i[1][3]),(0,255,0), 1)
    cv2.circle(im, (i[0][0], i[0][1]), 5, (0, 0, 0))
    cv2.line(im,(i[0][0],i[0][1]),(i[1][0],i[1][1]),(random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)), 2)
'''
#word1 DONEAppOpsPortal_DivRoutingName
filename = 'pics\\screenshotedImages\\startstation3.png'
im = cv2.imread(filename)
rectsFromSquares = retPotentials(im)
rectsFromOCR = testOCR(im)
for a in rectsFromOCR:
    rectsFromSquares.append(a)

rects=getAllCoupelesOnImage(im)

for i in rectsFromSquares:
    cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 1)
for i in rects:
    #cv2.rectangle(im, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 255, 0), 1)
    #cv2.rectangle(im, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 255, 0), 1)
    cv2.circle(im, (i[0][0], i[0][1]), 5, (0, 0, 0))
    cv2.line(im, (i[0][0], i[0][1]), (i[1][0], i[1][1]),(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)

cv2.imshow('image', im)
cv2.waitKey(0)