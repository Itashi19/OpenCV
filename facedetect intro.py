import cv2

cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#img=cv2.imread('test.jpg')
cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    faces=cascade.detectMultiScale(gray,1.1,3)
    #print(faces)

    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255),2)

    cv2.imshow('window',frame)


    if cv2.waitKey(1)==27:
        cv2.destroyAllWindows()
        break