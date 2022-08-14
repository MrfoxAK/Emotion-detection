from unittest import result
import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
     ret,frame = cam.read()
     result = DeepFace.analyze(frame,actions=['emotion'])

     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

     # draw a rectangle
     for(x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

     font = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(frame,
     result['dominant_emotion'],
     (50,50),
     font,3,
     (0,0,255),
     2,
     cv2.LINE_4)

     cv2.imshow("Video",frame)

     if cv2.waitKey(2) & 0xFF == ord('q'):
          break

cam.release()
cv2.destroyAllWindows()