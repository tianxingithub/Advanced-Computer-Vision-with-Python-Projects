import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon=0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode,self.maxHands,
        #                                 self.detectionCon,self.trackCon)
        # static_image_mode=False,
        #                max_num_hands=2,
        #                model_complexity=1,
        #                min_detection_confidence=0.5,
        #                min_tracking_confidence=0.5):

        self.hands = self.mpHands.Hands(max_num_hands=3)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # mediapipe.python.slution_base.SolutionOutputs
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPostion(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int (lm.y*h)
                # print(id, cx, cy) # id x y
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0),cv2.FILLED) # bgr

        return self.lmList

    def finersUp(self):
        fingers = []

        # Thumb, left hand is <
        # Thumb, right hand is >
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            # print("Index finger open")
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # print("Index finger open")
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPostion(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)  # 延迟的时间 1ms 0:暂停


if __name__ == "__main__":
    main()