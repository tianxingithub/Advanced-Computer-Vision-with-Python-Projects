import cv2
import time
import os
# import mediapipe as mp
import HandTrackingMoudule as htm

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("Videos/1.mp4")
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)


# print(len(overlayList))

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    _, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPostion(img, draw=False)
    # print(lmList)
    if(len(lmList)) != 0:
        fingers = []

        # Thumb, left hand is <
        # Thumb, right hand is >
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            # print("Index finger open")
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                # print("Index finger open")
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h,0:w] = overlayList[totalFingers]  # 0:200,0:200 height, width

        cv2.rectangle(img, (20,255), (170, 425), (0,255,255),cv2.FILLED)
        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)

    # FPS IFO
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}',(5,475),cv2.FONT_HERSHEY_PLAIN,
                2,(0,255,0),2)


    cv2.imshow("FingerCounting", img)
    cv2.waitKey(1)

