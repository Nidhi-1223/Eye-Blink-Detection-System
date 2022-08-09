import cv2
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
from scipy.spatial import distance as dist     #for calculating dist b/w the eye landmarks

#starting the video capture
cap = cv2.VideoCapture(0)
ret,img = cap.read()

#------defining a function to calulate the EAR-----------#
def calculate_EAR(eye) :
    #---calculate the verticle distances---#
    y1 = dist.euclidean(eye[1] , eye[5])
    y2 = dist.euclidean(eye[2] , eye[4])

    #----calculate the horizontal distance---#
    x1 = dist.euclidean(eye[0],eye[3])

    #----------calculate the EAR--------#
    EAR = (y1+y2) / x1
    return EAR

#---------Mark the eye landmarks-------#
def mark_eyeLandmark(img , eyes):
    for eye in eyes:
        pt1,pt2 = (eye[1] , eye[5])
        pt3,pt4 = (eye[0],eye[3])
        cv2.line(img,pt1,pt2,(200,00,0),2)
        cv2.line(img, pt3, pt4, (200, 0, 0), 2)
    return img

#---------Variables-------#
blink_thresh = 0.5
succ_frame = 2
count_frame = 0

#-------Eye landmarks------#
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']



#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/hp1/Documents/College/Coding/Good Projects/Eye-blink-detection-system/data/shape_predictor_68_face_landmarks.dat")

while(ret):
    ret,img = cap.read()
    #Coverting the recorded image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray,5,1,1)

    #---detecting the faces---#
    faces = detector(gray)
    print(faces)

    

    cv2.imshow("Webcam", img)

    cv2.putText(img, "look the program is working",  (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,3)



    key = cv2.waitKey(20)
    print(key)
    if(key == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()