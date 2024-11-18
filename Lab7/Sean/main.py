import cv2
import numpy as np
from calibration import calibration

# Object point
face_x = 15
face_y = 15
people_x = 110
people_y = 210

cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

# Calibration
# calibration(cap)
fs = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_READ)
intrinsic = fs.getNode("intrinsic").mat()
distortion = fs.getNode("distortion").mat()

while True:
    ret, frame = cap.read()
    people_rects, weights = hog.detectMultiScale(frame, 
                                          winStride=(8, 8),
                                          scale=1.1,
                                          useMeanshiftGrouping = False)
    face_rects = face_cascade.detectMultiScale(frame, 
                                               scaleFactor=1.06,
                                               minNeighbors=5,
                                               minSize=(60, 60))

    for (x, y, w, h) in people_rects:
        img_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        obj_pts = np.array([[0, 0, 0], [people_x, 0, 0], [people_x, people_y, 0], [0, people_y, 0]], dtype=np.float32)
        _, _, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsic, distortion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f'{tvec[2][0]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for (x, y, w, h) in face_rects:
        img_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        obj_pts = np.array([[0, 0, 0], [face_x, 0, 0], [face_x, face_y, 0], [0, face_y, 0]], dtype=np.float32)
        _, _, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsic, distortion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{tvec[2][0]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(10)