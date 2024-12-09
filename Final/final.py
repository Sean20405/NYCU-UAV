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
    max_ratio = 0
    # 遍歷每個九宮格
    for i, (h1, h2) in enumerate(zip(h_borders[:-1], h_borders[1:])):
        for j, (w1, w2) in enumerate(zip(w_borders[:-1], w_borders[1:])):
            # 取得當前格子的所有像素
            region = frame[h1:h2, w1:w2]
            # 計算黑色像素（值為0）的數量
            black_pixels = np.sum(region == 0)
            # 計算黑色像素的比例
            black_ratio = black_pixels / total_pixels
            max_ratio = max(max_ratio, black_ratio)
            
            # 根據位置設置對應的格子值
            pos = ['tl', 'tm', 'tr',
                  'ml', 'mm', 'mr',
                  'bl', 'bm', 'br'][i*3 + j]
            squares[pos] = 1 if black_ratio > threshold else 0
    
    # 返回結果列表，保持原有的順序
    return [squares['tl'], squares['tm'], squares['tr'],
            squares['ml'], squares['mm'], squares['mr'],
            squares['bl'], squares['bm'], squares['br']], max_ratio

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

        detected_squares, black_ratio = line_follower(gray)
        print(black_ratio)
        frame = cv2.putText(frame, text=f'battery: {drone.get_battery()}%', org=(600, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=1)
        frame = put_detected_square(frame, detected_squares, True)
        cv2.imshow("drone", frame)
        print(detected_squares)
        key = cv2.waitKey(50)
        if key != -1:
            keyboard(drone, key)
        else:
            lr, fb, ud, rot = speed_output
            if horizontal_trace and detected_squares[:3] == [1,1,1]:
                ud += 10
            elif horizontal_trace and  detected_squares[-3:] == [1,1,1]:
                ud -= 5
            elif not horizontal_trace and detected_squares[::3] == [1,1,1]:
                lr -= 5
            elif not horizontal_trace and detected_squares[2::3] == [1,1,1]:
                lr += 5
            fb += int(mss((0.4 - black_ratio) * 60, 20)) 
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
                cv2.destroyAllWindows()
                return
            else: 
                # continue
                # if abs(y_err) >= 10 or abs(x_err) >= 10:
                #     drone.send_rc_control(int(xv), 0, int(yv), 0)
                # else:
                drone.send_rc_control(int(xv), int(zv/1.5), int(yv), 0)
        else:
            if markId == 2:
                drone.send_rc_control(0, 0, 0, 0)
            else:
                drone.send_rc_control(0, 0, 0, 0)

# See the multiple markers with the same markId (using x coor to compare), follow the nearest one 
def see_multi(drone, markId):
    frame_read = drone.get_frame_read()
    # cap = cv2.VideoCapture(0)

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
        # ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIds)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)

        elif markerIds is not None:
            # Find the index of markId in markerIds
            target_idxes = []
            for i, id in enumerate(markerIds):
                if id[0] == markId:
                    target_idxes.append(i)
                    
            if not target_idxes:  # No target found
                continue

            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

            # Find the nearest marker and follow it
            target_idx = target_idxes[0]
            if type(target_idxes) == list:
                min_x_dist = float('Inf')
                for idx in target_idxes:
                    x_err = abs(tvec[idx][0][0])  ### TODO: calibration or using distance
                    if x_err < min_x_dist:
                        min_x_dist = x_err
                        target_idx = idx
                    
                    # Put x_err of each marker on the frame
                    text_coor = (np.sum(markerCorners[idx][0], axis=0) / 4).tolist()
                    text_coor = tuple([int(i) for i in text_coor])
                    text_coor = (text_coor[0], text_coor[1] + 25 * (i+1))
                    print(text_coor)    
                    cv2.putText(frame, text=f'idx: {idx}, x_err: {round(x_err, 2)}',
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                                org=text_coor, color=(0, 255, 0), thickness=1)

            # Display the nearest marker
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[target_idx], tvec[target_idx], 7)

            (x_err, y_err, z_err) = tvec[target_idx][0]
            z_err = z_err - 40
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
            zv = mss(z_err, 20)
            rv = mss(yaw_err)
            # print(xv, yv, zv, rv)
            # drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if abs(z_err) <= 10 and abs(y_err) <= 50 and abs(x_err) <= 50:
                print("Saw marker", markId)
                # cap.release()
                cv2.destroyAllWindows()
                return
            else: 
                # continue
                # if abs(y_err) >= 10 or abs(x_err) >= 10:
                #     drone.send_rc_control(int(xv), 0, int(yv), 0)
                # else:
                drone.send_rc_control(int(xv), int(zv//2), int(yv), 0)
                # print("Send speed", xv, yv, zv, rv)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            # print("Send speed", 0, 0, 0, 0)

        cv2.imshow('drone', frame)

def test():
    drone = Tello()
    drone.connect()
    drone.streamon()
    drone.takeoff()

    print("Moving down!")
    trace_line(drone, [0,0,-15,0], [0,1,0,1,1,0,0,0,0], False)
    print("7 corner detected")

    print("Moving left! Going through the table")
    trace_line(drone, [-20,0,0,0], [0,1,2,0,1,2,0,0,0], False)
    print("8 corner detected")

    print("Moving up!")
    trace_line(drone, [0,0,20,0], [0,0,0,1,1,0,0,1,0], False)
    print("9 corner detected")

    print("Moving left!")
    trace_line(drone, (-20,0,0,0), [0,2,2,0,1,1,0,0,0], True)
    print("10 corner detected")
    see(drone, 2)
    drone.rotate_clockwise(180)
    drone.move("forward", 70)
    detected_doll = detect_objects(drone)
    print(detected_doll)
    if detected_doll == "Kanahei":
        drone.move("left", 50)
    else:
        drone.move("right", 50)
    
    drone.move("up", 30)
    drone.move("forward", 100)
    see(drone, 3)
    drone.land()

def main():
    drone = Tello()
    drone.connect()
    drone.streamon()
    drone.takeoff()
    
    # 1. 飛到人臉前，飛過板子，看到第二張人臉，飛過桌子底下
    # see_face(drone, face_cascade)
    # drone.move("up", 75)
    # drone.move("forward", 130)
    # drone.move("down", 130)
    # see_face(drone, face_cascade)
    # drone.move("down", 70)
    # drone.move("forward", 180)

    # 2. 偵測娃娃，開始循線
    drone.move("up", 50)
    detected_doll = detect_objects(drone)
    print(f"Saw {detect_objects}\n")
    see(drone, 1)
    if detected_doll == "Kanahei":
        trace_line(drone, [0,0,15,0], [0,0,0,1,1,0,0,1,0], False)
        print("1 corner")
        trace_line(drone, [-15,0,0,0], [0,0,0,0,1,1,0,1,0], True)
        print("2 corner")
        trace_line(drone, [0,0,-15,0], [0,1,0,1,1,0,0,0,0], False)
        print("3 corner")
        trace_line(drone, [-15,0,0,0],  [0,1,2,0,1,2,0,0,0], False)
        print("4 corner")
        trace_line(drone, [0,0,15,0], [0,0,0,1,1,0,0,1,0], False)
        print("5 corner")
        trace_line(drone, [-15,0,0,0], [0,1,0,1,1,1,2,0,2], True)
        print("6 corner")
        trace_line(drone, (0,0,15,0), [0,0,0,1,1,0,0,1,0], False)
        print("7 corner")
        trace_line(drone, (-15,0,0,0), [0,0,0,0,1,1,0,1,0], True)
        print("8 corner")
        trace_line(drone, (0,0,-20,0), [0,1,0,1,1,1,0,0,0], False)
        print("9 corner")
        trace_line(drone, (-15,0,0,0), [0,2,2,0,1,1,0,0,0], True)
        print("9 corner")
    else:
        print("Moving up!")
        trace_line(drone, [0,0,15,0], [0,0,0,1,1,0,0,1,0], False)
        print("1 corner detected")

        print("Moving left!")
        trace_line(drone, [-15,0,0,0], [2,1,0,1,1,1,2,0,0], True)
        print("2 corner detected")

        print("Moving up!")
        trace_line(drone, (0,0,15,0), [0,0,0,1,1,0,0,1,0], False)
        print("3 corner detected")

        print("Moving left!")
        trace_line(drone, (-12,0,0,0), [0,2,2,0,1,1,0,1,2], True)
        print("4 corner detected")

        print("Moving down!")
        trace_line(drone, (0,0,-15,0), [0,1,0,1,1,1,0,0,0], False)
        print("5 corner detected")

        print("Moving left!")
        trace_line(drone, (-10,0,0,0), [0,0,0,0,1,1,0,1,0], True)
        print("6 corner detected")

        print("Moving down!")
        trace_line(drone, [0,0,-15,0], [0,1,0,1,1,0,0,0,0], False)
        print("7 corner detected")

        print("Moving left! Going through the table")
        trace_line(drone, [-15,0,0,0],  [0,1,2,0,1,2,0,0,0], True)
        print("8 corner detected")

        print("Moving up!")
        trace_line(drone, [0,0,20,0], [0,0,0,1,1,0,0,1,0], False)
        print("9 corner detected")

        print("Moving left!")
        trace_line(drone, (-20,0,0,0), [0,2,2,0,1,1,0,0,0], True)
        print("10 corner detected")

    # 結束循線，判斷娃娃決定路徑，準備降落
    see(drone, 2)
    drone.rotate_clockwise(180)
    drone.move("down", 30)
    detected_doll = detect_objects(drone)
    if detected_doll == "Kanahei":
        drone.move("left", 50)
    else:
        drone.move("right", 50)
    drone.move("up", 30)
    drone.move("forward", 180)
    see_multi(drone, 3)
    drone.land()
    
if __name__ == "__main__":
    # test()
    main()
