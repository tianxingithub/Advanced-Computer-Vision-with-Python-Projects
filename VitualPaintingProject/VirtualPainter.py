import cv2
import numpy as np
import time
import os
import HandTrackingMoudule as htm


###########################
brushThickness = 5
eraserThickness = 50
###########################

folderPath = "Header"
myList = os.listdir(folderPath)

# print(myList)

overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColor = (255,255,255)


cap = cv2.VideoCapture(0)
# cap.set(3, 800)
# cap.set(4, 600)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # ideal
    # 1. Import image
    _, img = cap.read()
    img = cv2.flip(img, 1) # 取消镜像

    # 2. Find hand landmarks
    img = detector.findHands(img,draw=False)
    lmList = detector.findPostion(img,draw=False)

    if len(lmList) != 0:

        # print(lmList)


        # tip of index and minddle fingers
        x1, y1 = lmList[8][1:] # 1: - one to end
        x2, y2 = lmList[12][1:] # 1: - one to end


        # 3. Check which fingers are up (index)

        fingers = detector.finersUp()
        # print(fingers)
        # 4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # cv2.rectangle(img, (x1, y1-15), (x2, y2+15), drawColor, cv2.FILLED)
            # print("Slection Mode")
            # Cheking for the lick
            temh = 64
            if y1 < 64:
                if 190-temh < x1 < 190:
                    header = overlayList[1]
                    drawColor = (0,128,255)
                elif 321-temh < x1 < 321:
                    header = overlayList[2]
                    drawColor = (255,102,178)
                elif 450-temh < x1 < 450:
                    header = overlayList[3]
                    drawColor = (128, 255, 0)
                elif 515 < x1 < 620:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                    # brushThickness = eraserThickness
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 5,drawColor, cv2.FILLED)
            # print("Draw Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    h,w,c=header.shape
    img[0:h,0:w] = header  # img[h, w]
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Painting", img)
    # cv2.imshow("imgCanvas", imgCanvas)
    # cv2.imshow("imgIvc", imgInv)
    cv2.waitKey(1)











