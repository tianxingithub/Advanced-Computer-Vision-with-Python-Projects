import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=4, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

    # def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2) # max_num_faces=1 rgb
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces, False,
                                                 self.minDetectionCon ,self.minTrackCon )
        self.drawSpec= self.mpDraw.DrawingSpec((255,0,0),thickness=1,circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)  # FACE_CONNECTIONS
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.1, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

    def drawPoints(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec,self.drawSpec) # FACE_CONNECTIONS
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    print(id, x, y)


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("Videos/22.mp4")
    facemesh = FaceMeshDetector(maxFaces=3)
    pTime = 0
    while True:
        suceess, img = cap.read()
        # img, faces = facemesh.findFaceMesh(img,False)
        img, faces = facemesh.findFaceMesh(img,True)
        if len(faces) != 0:
            print(faces[0])
        # facemesh.drawPoints(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.imshow("Image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()