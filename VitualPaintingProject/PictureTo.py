import cv2

path = r'E:\_Files\fallSun.jpeg'
imgSun = cv2.imread(path)
# if imgSun != 0 :
#     print("read OK")
# else:
#     print("read not OK")
cv2.imshow('img',imgSun)
cv2.waitKey(0)