import cv2
import matplotlib.image as mpling
import matplotlib.pyplot as plt
import numpy as np


img=cv2.imread('1.jpg',0)
img =cv2.medianBlur(img,5)#filtre pour minimiser le bruit 
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)# A VERIFIER
circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,2,param1=200,param2=60,minRadius=20,maxRadius=100)
circles=np.uint16(np.around(circles))

for i in circles[0,:]:
    #dessiner le contour et le centre
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg)
plt.show()
