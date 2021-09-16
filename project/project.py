import cv2
import numpy as np
img1=cv2.imread('imager/peanut.jpg',0)
img2=cv2.imread('trainimg\plastic container.jpg',0)

orb=cv2.ORB_create(nfeatures=1000)

kp1,des1=orb.detectAndCompute(img1,None) 
kp2,des2=orb.detectAndCompute(img2,None) 
brf=cv2.BFMatcher()
matches=brf.knnMatch(des1,des2,k=2)

good=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#imgkp1=cv2.drawKeypoints(img,kp1,None)
#cv2.imshow('kp1',imgkp1)
cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
cv2.imshow('image3',img3)
cv2.waitKey(0)