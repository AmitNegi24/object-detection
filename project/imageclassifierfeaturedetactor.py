import cv2
import numpy as np
import os
# Assigning all the files/images to the program 
path='imager'
orb=cv2.ORB_create(nfeatures=1000)

images=[]
classNames=[]
mylist=os.listdir(path)
print( mylist)
print('Total file detected=',len(mylist))

for cls in mylist: 
    imgcur=cv2.imread(f'{path}/{cls}',0) #current image
    images.append(imgcur)
    classNames.append(os.path.splitext(cls)[0]) #removing the dot format of images like .jpg
print(classNames)

def findDes(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def FindId(img,desList,thrshold=12):
    kp2,des2=orb.detectAndCompute(img,None)
    brf=cv2.BFMatcher()
    matchlistgood=[]
    lastVal = -1
    try:
        for des in desList:
            matches=brf.knnMatch(des, des2, k=2)
            good=[]
            for m,n in matches:
             if m.distance < 0.75*n.distance:
                 good.append([m])
            matchlistgood.append(len(good))
    except:
        pass
    #print(matchlistgood)
    if len(matchlistgood)!= 0:
        if max(matchlistgood) > thrshold:
            lastVal= matchlistgood.index(max(matchlistgood))
    return lastVal

desList=findDes(images)  
print(len(desList))      

cam=cv2.VideoCapture(0)

while True:
    success, img2= cam.read()
    imgoriginal= img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    id = FindId(img2,desList)   #calling of FindId function 3rd argument is automatically i.e threshold=12
    print(id)
    if id != -1:
        cv2.putText(imgoriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    cv2.imshow('img2',imgoriginal)
    cv2.waitKey(1) 



# img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# #imgkp1=cv2.drawKeypoints(img,kp1,None)
# #cv2.imshow('kp1',imgkp1)
# cv2.imshow('image1',img1)
# cv2.imshow('image2',img2)
# cv2.imshow('image3',img3)
# cv2.waitKey(0)