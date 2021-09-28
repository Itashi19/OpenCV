import cv2
import numpy as np

img=cv2.imread('dog.jpg') #to convert in grayscale apply 0

#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#to blur the image
# 1.img=cv2.blur(img,(2,2))
#2 to blur
#img=cv2.GaussianBlur(img,(5,5),10)

#thresholding/binarizing image
#convert img to grayscale
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#_,img=cv2.threshold(img,25,255,cv2.THRESH_BINARY)

#img2=cv2.erode(img,np.ones((5,5)))

#crop the image
img=img[80:220,210:410]


while True:
    cv2.imshow('original',img)
    #cv2.imshow('eroded', img2)

    if cv2.waitKey(0)==27:
        cv2.destroyAllWindows()
        break
