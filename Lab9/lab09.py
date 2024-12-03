import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard


black_thres = 30

sq = {
    "tl":0, "tm":1, "tr":2, "ml":3, "mm":4, "mr":5, "bl":6, "bm":7, "br":8
}

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
    
    # 遍歷每個九宮格
    for i, (h1, h2) in enumerate(zip(h_borders[:-1], h_borders[1:])):
        for j, (w1, w2) in enumerate(zip(w_borders[:-1], w_borders[1:])):
            # 取得當前格子的所有像素
            region = frame[h1:h2, w1:w2]
            # 計算黑色像素（值為0）的數量
            black_pixels = np.sum(region == 0)
            # 計算黑色像素的比例
            black_ratio = black_pixels / total_pixels
            print(black_ratio)
            
            # 根據位置設置對應的格子值
            pos = ['tl', 'tm', 'tr',
                  'ml', 'mm', 'mr',
                  'bl', 'bm', 'br'][i*3 + j]
            squares[pos] = 1 if black_ratio > threshold else 0
    
    # 返回結果列表，保持原有的順序
    return [squares['tl'], squares['tm'], squares['tr'],
            squares['ml'], squares['mm'], squares['mr'],
            squares['bl'], squares['bm'], squares['br']]
# def line_follower(frame):
#     height, width = frame.shape
#     width_1_4 = int(width/4)
#     width_mid = int(width/2)
#     width_3_4 = 3 * width_1_4
#     height_1_4 = int(height/4)
#     threshold = 0.2
#     detect_thick = 6
#     tl = tm = tr = ml = mm = mr = bl = bm = br = 0

#     for i in range(height):
#         for j in range(width_mid-detect_thick, width_mid):
#             if frame[i, j] == 0:
#                 if(i < height/3):
#                     tm += 1
#                 elif(i < height/3*2):
#                     mm += 1
#                 else:
#                     bm += 1
#         for j in range(width_1_4-detect_thick, width_1_4):
#             if frame[i, j] == 0:
#                 if(i < height/3):
#                     tl += 1
#                 elif(i < height/3*2):
#                     ml += 1
#                 else:
#                     bl += 1
#         for j in range(width_3_4-detect_thick, width_3_4):
#             if frame[i, j] == 0:
#                 if(i < height/3):
#                     tr += 1
#                 elif(i < height/3*2):
#                     mr += 1
#                 else:
#                     br += 1

#     tl = 1 if tl > height * threshold else 0
#     tm = 1 if tm > height * threshold else 0
#     tr = 1 if tr > height * threshold else 0
#     ml = 1 if ml > height * threshold else 0
#     mm = 1 if mm > height * threshold else 0
#     mr = 1 if mr > height * threshold else 0
#     bl = 1 if bl > height * threshold else 0
#     bm = 1 if bm > height * threshold else 0
#     br = 1 if br > height * threshold else 0

#     return [tl, tm, tr, ml, mm, mr, bl, bm, br]

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

def trace_line(drone, speed_output, target_square):
    detected_squares = [0,0,0,0,0,0,0,0,0]
    while detected_squares[target_square] == 0:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)

        detected_squares = line_follower(gray)

        frame = cv2.putText(frame, text=f'battery: {drone.get_battery()}%', org=(600, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=1)
        frame = put_detected_square(frame, detected_squares, True)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(50)
        if key != -1:
            keyboard(drone, key)
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

    fs = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_READ)
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

        cv2.imshow('frame', frame)
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
            z_err = z_err - 30
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
                drone.send_rc_control(0, -3, 0, 0)
            else:
                drone.send_rc_control(0, 0, 0, 0)

if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()
    
    see(drone, 2)
    drone.move("right", 40)
    # drone.move('right', 0)
    trace_line(drone, (0,0,5,0), sq["mr"])
    print(1)
    trace_line(drone, (5,0,0,0), sq["tm"])
    print(2)
    trace_line(drone, (0,0,5,0), sq["mr"])
    print(3)
    trace_line(drone, (5,0,0,0), sq["bm"])
    print(4)
    drone.land()
    
    while False:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)
        detected_squares = line_follower(gray)


        frame = cv2.putText(frame, text=f'battery: {drone.get_battery()}%', org=(800, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=1)
        frame = put_detected_square(frame, detected_squares, True)
        cv2.imshow("drone", frame)

        # frame = cv2.putText(gray, text=f'battery: {drone.get_battery()}%', org=(800, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=1)
        # frame = put_detected_square(gray, detected_squares)
        # cv2.imshow("drone", gray)

        # print(frame)
        # print(frame.shape.len)
        # print(detected_squares)
        key = cv2.waitKey(50)