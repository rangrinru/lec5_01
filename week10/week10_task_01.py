import cv2
import numpy as np

xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml_path)

threshold_move = 50
diff_compare = 10

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, img = cap.read()
    #img_third = cv2.flip(img_third, 0)
    scr = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_gray = face_cascade.detectMultiScale(img_gray)      
    
    for (x, y, w, h) in faces_gray:
        scr = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('Motion detection', scr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()






