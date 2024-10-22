import cv2
import numpy as np
import keyboard
from djitellopy import Tello
import keyboard_djitellopy
from pyimagesearch.pid import PID

def detection(frame_read):   
    print("Detection...")
    # cap = cv2.VideoCapture(0)
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Get param of camera calibration
    fs = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    while True:
        frame = frame_read.frame
        # ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if markerIds is None:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)
            continue
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # Pose estimation for single markers. 
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.15, intrinsic, distortion)
        x, y, z = tvec[0][0]
        text_coor = (np.sum(markerCorners[0][0], axis=0) / 4).tolist()
        print(text_coor)
        text_coor = tuple([int(i+20) for i in text_coor])
        # print(x, y, z)
        print(text_coor)
        # print(markerCorners[0][0])
        cv2.putText(frame, text=f'x: {round(x, 2)}  y: {round(y, 2)}  z: {round(z, 2)}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, org=text_coor, color=(0, 255, 255), thickness=1)
        frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 7)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)

def track_pattern(frame_read, x_pid, y_pid, z_pid, yaw_pid, fs, intrinsic, distortion):
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # cap = cv2.VideoCapture(0)

    while True:
        # Control the drone with keyboard
        key = cv2.waitKey(33)
        if key != -1:
            keyboard_djitellopy.telloKeyboard(drone, key)

        # Find the marker
        frame = frame_read.frame
        # ret, frame = cap.read()

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if markerIds is None:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)
            continue
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # Pose estimation for single markers. 
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.15, intrinsic, distortion)
        x, y, z = tvec[0][0]
        # rmat, _ = cv2.Rodrigues(rvec)

        # Control back and forth
        print(x, y, z)
        z_error = (z - 1) * 20 * 1.5  # unit: m
        z_update = z_pid.update(z_error, sleep=0)
        z_update = int(np.clip(z_update, -20, 20))
        print("z: ", z_error, z_update)

        # Control right and left
        x_error = (x+0.13) * 50 * 1.5
        x_update = x_pid.update(x_error, sleep=0)
        x_update = int(np.clip(x_update, -20, 20))
        print("x: ", x_error, x_update)
        
        # Control up and down
        y_error = (y+0.34) * 50 * 1.5
        y_update = y_pid.update(y_error, sleep=0)
        y_update = int(np.clip(y_update, -20, 20))
        print("y: ", y_error, y_update)

        # Control yaw
        yaw_error = rvec[0][0][2] * 5 * 1.5
        yaw_update = yaw_pid.update(yaw_error, sleep=0)
        yaw_update = int(np.clip(yaw_update, -20, 20))
        print("yaw: ", yaw_error, yaw_update)

        # if max(x_update, y_update, z_update, yaw_update) == x_update:
        #     drone.send_rc_control(x_update, 0, 0, 0)
        # elif max(x_update, y_update, z_update, yaw_update) == y:
        #     drone.send_rc_control(0, 0, y_update, 0)
        # elif max(x_update, y_update, z_update, yaw_update) == z:
        #     drone.send_rc_control(0, z_update, 0, 0)
        # else:
        #     drone.send_rc_control(0, 0, 0, yaw_update)
            


        cv2.imshow('frame', frame)


if __name__ == '__main__':
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)

    global is_flying
    # Get param of camera calibration
    fs = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    x_pid.initialize()
    y_pid.initialize()
    z_pid.initialize()
    yaw_pid.initialize()

    drone.streamon()
    frame_read = drone.get_frame_read()
    track_pattern(frame_read, x_pid, y_pid, z_pid, yaw_pid, fs, intrinsic, distortion)
    track_pattern(0, x_pid, y_pid, z_pid, yaw_pid, fs, intrinsic, distortion)