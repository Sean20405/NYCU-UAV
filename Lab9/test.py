import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

def line_follower(frame):
    height, width = frame.shape
    width_1_4 = int(width/4)
    width_mid = int(width/2)
    width_3_4 = 3 * width_1_4
    height_1_4 = int(height/4)
    height_mid = int(height/2)
    height_3_4 = 3 * height_1_4
    threshold = 0.2
    tl = tm = tr = ml = mm = mr = bl = bm = br = 0

    #h_top, h_mid, h_bottom = 0, 0, 0
    for i in range(height):
        for j in range(width_mid-3, width_mid):
            if frame[i, j] == 0:
                if(i < height/3):
                    tm += 1
                elif(i < height/3*2):
                    mm += 1
                else:
                    bm += 1
        for j in range(width_1_4-3, width_1_4):
            if frame[i, j] == 0:
                if(i < height/3):
                    tl += 1
                elif(i < height/3*2):
                    ml += 1
                else:
                    bl += 1
        for j in range(width_3_4-3, width_3_4):
            if frame[i, j] == 0:
                if(i < height/3):
                    tr += 1
                elif(i < height/3*2):
                    mr += 1
                else:
                    br += 1

    # w_left, w_mid, w_right = 0, 0, 0
    # for i in range(width):
    #     for j in range(height_mid-3, height_mid):
    #         if frame[j, i] == 0:
    #             if(i < width/3):
    #                 w_left += 1
    #             elif(i < width/3*2):
    #                 w_mid += 1
    #             else:
    #                 w_right += 1

    tl = 1 if tl > height * threshold else 0
    tm = 1 if tm > height * threshold else 0
    tr = 1 if tr > height * threshold else 0
    ml = 1 if ml > height * threshold else 0
    mm = 1 if mm > height * threshold else 0
    mr = 1 if mr > height * threshold else 0
    bl = 1 if bl > height * threshold else 0
    bm = 1 if bm > height * threshold else 0
    br = 1 if br > height * threshold else 0
    # if h_top > height * 3 * threshold * 0.333:
    #     if w_left > width * 3 * threshold * 0.333:
    #         tl = 1
    #     if w_mid > width * 3 * threshold * 0.333:
    #         tm = 1
    #     if w_right > width * 3 * threshold * 0.333:
    #         tr = 1
    # if h_mid > height * 3 * threshold * 0.333:
    #     if w_left > width * 3 * threshold * 0.333:
    #         ml = 1
    #     if w_mid > width * 3 * threshold * 0.333:
    #         mm = 1
    #     if w_right > width * 3 * threshold * 0.333:
    #         mr = 1
    # if h_bottom > height * 3 * threshold * 0.333:
    #     if w_left > width * 3 * threshold * 0.333:
    #         bl = 1
    #     if w_mid > width * 3 * threshold * 0.333:
    #         bm = 1
    #     if w_right > width * 3 * threshold * 0.333:
    #         br = 1

    return [tl, tm, tr, ml, mm, mr, bl, bm, br]

def put_detected_square(frame, detected_squares, is_gray):
    height, width = 0, 0
    if is_gray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
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


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    detected_squares = line_follower(gray)
        
    frame = put_detected_square(frame, detected_squares, False)
    cv2.imshow("drone", frame)

    # print(frame)
    # print(frame.shape.len)
    # print(detected_squares)
    key = cv2.waitKey(50)
