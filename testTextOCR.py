from PIL import Image
import pytesseract
import cv2
import os
#import tesseract

def isYinList(y, list):
    if (len(list) == 0):
        return False
    for i in range(len(list)):
        if abs(y - list[i]) <= 10:
            return True
    return False


def isYcloseToAnumInList(y, num):
    if abs(y - num) <= 10:
        return True
    return False


def lowestIndexinList(index, list):
    lowest = list[0][index]
    for smallList in list:
        if (int)(smallList[index]) < (int)(lowest):
            lowest = smallList[index]
    return int(lowest)


def biggestIndexinList(index, list):
    biggest = 0
    for smallList in list:
        if (int)(smallList[index]) > (int)(biggest):
            biggest = smallList[index]
    return int(biggest)


#pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\{}\\AppData\\Local\\Tesseract-OCR\\tesseract".format(os.getlogin())
#pytesseract.pytesseract.tesseract_cmd="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\{}\\AppData\\Local\\Tesseract-OCR\\tesseract".format(os.getlogin())

def testOCR(fileName):
    # read the image and get the dimensions
    #img = cv2.imread(fileName)
    img=fileName
    #img = img[90:-80, :]
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w, _ = img.shape  # assumes color image

    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(img2)  # also include any config options you use
    boxes = boxes.splitlines()

    words = []
    letterSize = 10
    for i in range(len(boxes)):
        num = boxes[i].split(' ')
        if (ord(num[0]) >= 64 and ord(num[0]) <= 90) or (ord(num[0]) >= 97 and ord(num[0]) <= 122) or (ord(num[0]) >= 33 and ord(num[0]) <= 59):
            if len(words) == 0:
                words.append([])
                words[0].append(num)
                if (ord(num[0]) >= 32 and ord(num[0]) <= 46):
                    letterSize = 3 * abs(int(num[3]) - int(num[1]))
                else:
                    letterSize = 1.3 * abs(int(num[4]) - int(num[2]))
            elif abs(int(words[-1][-1][3]) - int(num[1])) <= letterSize and ord(words[-1][-1][0]) != 58:  # abs(int(words[-1][-1][4])-int(num[4]))<=abs(int(words[-1][-1][4])-int(words[-1][-1][2])) and abs(int(words[-1][-1][3])-int(num[1]))<=letterSize:
                words[-1].append(num)
                letterSize = (abs(int(num[4]) - int(num[2])) + letterSize) / 2
            else:
                words.append([])
                words[-1].append(num)
                if (ord(num[0]) >= 32 and ord(num[0]) <= 46):
                    letterSize = 3 * abs(int(num[3]) - int(num[1]))
                else:
                    letterSize = 1.3 * abs(int(num[4]) - int(num[2]))
    listOfLists = []
    # sortedBoxes=sorted(boxes, key=lambda b: b.split(' ')[2])   # sort by age
    # draw the bounding boxes on the image
    # print(boxes)
    rects = []
    for i in range(len(words)):
        smallestX = lowestIndexinList(1, words[i])
        smallestY = h - lowestIndexinList(2, words[i])
        biggestX = biggestIndexinList(3, words[i])
        biggestY = h - biggestIndexinList(4, words[i])
        #img = cv2.rectangle(img, (smallestX, smallestY), (biggestX, biggestY), (0, 255, 0), 1)
        if(smallestX!=biggestX and smallestY!=biggestY):
            rects.append([smallestX, biggestY, biggestX, smallestY])
    '''for b in boxes:
       b = b.split(' ')
       img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 0, 255), 1)'''

    # show annotated image and wait for keypress
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    return rects

#im = cv2.imread('pics\\screenshotedImages\\DONEpitCreateTicket2.png')
#testOCR(im)
