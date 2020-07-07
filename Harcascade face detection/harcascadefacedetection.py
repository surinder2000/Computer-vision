import cv2

cap = cv2.VideoCapture(0)

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    status,photo = cap.read()
    face_coordinates = face_model.detectMultiScale(photo)
    if len(face_coordinates) != 0:
        x1 = face_coordinates[0][0]
        y1 = face_coordinates[0][1]
        x2 = x1 + face_coordinates[0][2]
        y2 = y1 + face_coordinates[0][3]
        photo = cv2.rectangle(photo, (x1,y1),(x2,y2),[0,0,255],5)
        cv2.imshow('Face Detection',photo)
        if cv2.waitKey(10) == 13:
            break
    else:
        pass
cv2.destroyAllWindows()
cap.release()
    