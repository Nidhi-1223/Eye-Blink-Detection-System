import numpy as np
import cv2
from skimage import io
face_cascade = cv2.CascadeClassifier('/home/hp1/Documents/College/Coding/Good Projects/Eye-blink-detection-system/python files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/hp1/Documents/College/Coding/Good Projects/Eye-blink-detection-system/python files/haarcascade_eye_tree_eyeglasses.xml')

first_read = True
# url = 'http://192.168.43.91/capture'
cap = cv2.VideoCapture(0)    
ret,img = cap.read()
while True:
    # img = io.imread(url)
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,1,1)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,scaleFactor=1.5,minNeighbors=5,minSize=(30,30))
            #minNeighbors value is often used to detect false positive

            if(len(eyes)>=2):
                if(first_read):
                    cv2.putText(img, "press s to begin", (100,100), cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                else:
                    print("----------------------")
            else:
                if(first_read):
                    cv2.putText(img, "No eyes detected", (100,100), cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                else:
                    print(" you loose")
                    first_read=True
    else:
        cv2.putText(img,"No face detected",(100,100),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)


    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        first_read = False


cap.release()
cv2.destroyAllWindows()
