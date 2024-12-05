import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from face_detection import see_face
from object_detection import detect_objects


black_thres = 30

sq = {
    "tl":0, "tm":1, "tr":2, "ml":3, "mm":4, "mr":5, "bl":6, "bm":7, "br":8
}


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def square_same(input, truth):
    for i in range(9):
        if truth[i] == 2:
            input[i] = 2
    return input == truth

def line_follower(frame):
    frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    height, width = frame.shape
    
    # 計算九宮格的邊界
    h_borders = [0, height//3, 2*height//3, height]
    w_borders = [0, width//3, 2*width//3, width]
    
    # 初始化九宮格的計數器
    squares = {
        'tl': 0, 'tm': 0, 'tr': 0,
        'ml': 0, 'mm': 0, 'mr': 0,
        'bl': 0, 'bm': 0, 'br': 0
    }
    
    # 計算每個格子的總像素數
    total_pixels = (height//3) * (width//3)
    threshold = 0.1  # 可以調整這個閾值
    back = False
    # 遍歷每個九宮格
    for i, (h1, h2) in enumerate(zip(h_borders[:-1], h_borders[1:])):
        for j, (w1, w2) in enumerate(zip(w_borders[:-1], w_borders[1:])):
            # 取得當前格子的所有像素
            region = frame[h1:h2, w1:w2]
            # 計算黑色像素（值為0）的數量
            black_pixels = np.sum(region == 0)
            # 計算黑色像素的比例
            black_ratio = black_pixels / total_pixels
            if (black_ratio > 0.5):
                back = True
            
            # 根據位置設置對應的格子值
            pos = ['tl', 'tm', 'tr',
                  'ml', 'mm', 'mr',
                  'bl', 'bm', 'br'][i*3 + j]
            squares[pos] = 1 if black_ratio > threshold else 0
    
    # 返回結果列表，保持原有的順序
    return [squares['tl'], squares['tm'], squares['tr'],
            squares['ml'], squares['mm'], squares['mr'],
            squares['bl'], squares['bm'], squares['br']], back

def put_detected_square(frame, detected_squares, is_gray):
    height, width = 0, 0
    if is_gray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)
        height, width = gray.shape
        frame = gray
    else:
        (height, width, _) = frame.shape

    w_mid = int(width/2)
    h_mid = int(height/2)

    x_list = [10,w_mid,width-100]
    y_list = [10,h_mid,height-10]

    for i, detected in enumerate(detected_squares):
        x = x_list[i % 3]
        y = y_list[int(i / 3)]
        if detected:
            cv2.putText(frame, text=f'black', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            # fontScale=0.7, org=(x, y), color=(0, 0, 0), thickness=1)
            fontScale=0.7, org=(x, y), color=(255, 255, 255), thickness=1)
        else:
            cv2.putText(frame, text=f'white', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            # fontScale=0.7, org=(x, y), color=(255, 255, 255), thickness=1)
            fontScale=0.7, org=(x, y), color=(0, 0, 0), thickness=1)
        
    return frame

def trace_line(drone, speed_output, target_square, horizontal_trace):
    detected_squares = [0,0,0,0,0,0,0,0,0]
    while not square_same(detected_squares, target_square):
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)

        detected_squares, back = line_follower(gray)

        frame = cv2.putText(frame, text=f'battery: {drone.get_battery()}%', org=(600, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=1)
        frame = put_detected_square(frame, detected_squares, True)
        cv2.imshow("drone", frame)
        print(detected_squares)
        key = cv2.waitKey(50)
        if key != -1:
            keyboard(drone, key)
        elif back:
            drone.send_rc_control(0,-5,0,0)
        else:
            if horizontal_trace and detected_squares[:3] == [1,1,1]:
                drone.send_rc_control(0,0,5,0)
            elif horizontal_trace and  detected_squares[-3:] == [1,1,1]:
                drone.send_rc_control(0,0,-5,0)
            elif not horizontal_trace and detected_squares[::3] == [1,1,1]:
                drone.send_rc_control(-5,0,0,0)
            elif not horizontal_trace and detected_squares[2::3] == [1,1,1]:
                drone.send_rc_control(5,0,0,0)
            else:
                lr, fb, ud, rot = speed_output
                drone.send_rc_control(lr, fb, ud, rot)
    drone.send_rc_control(0,0,0,0)

def mss(update, max_speed_threshold=30):
    if update > max_speed_threshold:
        update = max_speed_threshold
    elif update < -max_speed_threshold:
        update = -max_speed_threshold

    return update

def see(drone, markId):
    frame_read = drone.get_frame_read()

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("utils/calibrate.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()

    while True:
        frame = frame_read.frame
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIds)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.imshow('drone', frame)
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        elif markerIds is not None:
            # Find the index of markId in markerIds
            target_idx = None
            for i, id in enumerate(markerIds):
                if id[0] == markId:
                    target_idx = i
            if target_idx is None:
                continue

            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            (x_err, y_err, z_err) = tvec[target_idx][0]
            z_err = z_err - 50
            x_err = x_err * 2
            y_err = - (y_err + 10) * 2

            R, err = cv2.Rodrigues(np.array([rvec[target_idx]]))
            # print("err:", err)
            V = np.matmul(R, [0, 0, 1])
            rad = math.atan(V[0]/V[2])
            deg = rad / math.pi * 180
            # print(deg)
            yaw_err = yaw_pid.update(deg, sleep=0)
            
            x_err = x_pid.update(x_err, sleep=0)
            y_err = y_pid.update(y_err, sleep=0)
            z_err = z_pid.update(z_err, sleep=0)
            yaw_err = yaw_pid.update(yaw_err, sleep=0)

            print("errs:", x_err, y_err, z_err, yaw_err)
            
            xv = mss(x_err)
            yv = mss(y_err)
            zv = mss(z_err)
            rv = mss(yaw_err)
            # print(xv, yv, zv, rv)
            # drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if abs(z_err) <= 10 and abs(y_err) <= 50 and abs(x_err) <= 50:
                print("Saw marker", markId)
                return
            else: 
                # continue
                # if abs(y_err) >= 10 or abs(x_err) >= 10:
                #     drone.send_rc_control(int(xv), 0, int(yv), 0)
                # else:
                drone.send_rc_control(int(xv), int(zv//2), int(yv), 0)
        else:
            if markId == 2:
                drone.send_rc_control(0, 0, 0, 0)
            else:
                drone.send_rc_control(0, 0, 0, 0)

if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()
    # drone.takeoff()
    # time.sleep(5)
    # see_face(drone, face_cascade)
    # drone.move("up", 75)
    # drone.move("forward", 130)
    # drone.move("down", 130)
    # see_face(drone, face_cascade)
    # drone.move("down", 70)
    # drone.move("forward", 180)
    detected_doll = detect_objects(drone)
    print(detected_doll)
    drone.land()
    