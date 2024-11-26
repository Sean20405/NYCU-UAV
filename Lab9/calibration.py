import cv2
import numpy as np
from djitellopy import Tello

drone = Tello()
drone.connect()
drone.streamon()

object_point = np.zeros((9*6,3), np.float32)
for i in range(6):
    for j in range(9):
        object_point[i*9+j] = (i, j, 0)

object_points = []
img_points = []
frame_count = 0

while frame_count < 20:
    frame = drone.get_frame_read()
    frame = frame.frame
    h, w = frame.shape[:2]
    cv2.imshow("drone", frame)
    # cv2.imwrite("output/frame.jpg", frame)
    cv2.waitKey(33)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(frame_gray, (9, 6), None)
    if ret:
        frame_count += 1
        subcorner = cv2.cornerSubPix(frame_gray, corner, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(object_point)
        img_points.append(subcorner)
        drawn_frame = cv2.drawChessboardCorners(frame, (9, 6), subcorner, ret)
        # cv2.imwrite("drawn_frame.jpg", drawn_frame)
        cv2.waitKey(33)

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, (h, w), None, None)
f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)
f.write("distortion", distCoeffs)
f.release()