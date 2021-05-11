import numpy as np 
import cv2 
import pickle



face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open ("labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print (x,y,w,h)

        roi_gray=gray[y:y+h, x:x+w] # ycord_start,ycord_end xcord_start,xcord_end 
        roi_color=frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 :
            print (id_)
            print(labels[id_])

        #img_item="img-item.png"
        #cv2.imwrite(img_item, roi_gray)

        color=(0,255,0)
        stroke=2
        endcord_X=x+w
        endcord_Y=y+h
        cv2.rectangle(frame,(x,y),(endcord_X,endcord_Y),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

