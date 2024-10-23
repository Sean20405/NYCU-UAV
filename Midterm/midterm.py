from random import randint
from djitellopy import Tello
from pyimagesearch.pid import PID
import cv2
import numpy as np
import math

error = {1: [50, 50, 10],
         2: [50, 50, 10],
         3: [10, 10, 10], 
         4: [40, 40, 10], 
         5: [20, 20, 10], 
         6: [5, 10, 5]}

y_dist = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 20}
z_dist = {0: 75, 1: 75, 2: 75, 3: 75, 4: 75, 5: 75, 6: 240}

def keyboard(self, key):
    #global is_flying
    print("key:", key)
    # key = ord(key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    if key == ord('1'):
        self.takeoff()
        #is_flying = True
    if key == ord('2'):
        self.land()
        #is_flying = False
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print (battery)

def mss(update, max_speed_threshold=30):
    if update > max_speed_threshold:
        update = max_speed_threshold
    elif update < -max_speed_threshold:
        update = -max_speed_threshold

    return update

def teleop(drone):
    key = cv2.waitKey(100)
    drone.takeoff()
    key = cv2.waitKey(100)
    drone.move("forward", 50)
    key = cv2.waitKey(100)
    drone.move("back", 50)
    key = cv2.waitKey(100)
    drone.land()



def hasMarker(markerIds, stopId):
    for i, id in enumerate(markerIds):
        if id[0] == stopId and len(markerIds) == 1:
            print("STOPPPPP")
            return True
    return False

def see(drone, markId):
    frame_read = drone.get_frame_read()

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("param-drone.xml", cv2.FILE_STORAGE_READ)
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

        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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

            cv2.putText(frame, text=f'x: {round(x_err, 2)}  y: {round(y_err, 2)}  z: {round(z_err, 2)}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.4, org=(10, 10), color=(0, 255, 255), thickness=1)

            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            z_err = z_err - z_dist[markId]; z_err = z_err
            x_err = x_err
            y_err = - (y_err + y_dist[markId]) * 2

            R, err = cv2.Rodrigues(np.array([rvec[target_idx]]))
            # print("err:", err)
            V = np.matmul(R, [0, 0, 1])
            rad = math.atan(V[0]/V[2])
            deg = rad / math.pi * 180
            deg *= 2
            # print(deg)
            
            x_err = x_pid.update(x_err, sleep=0)
            y_err = y_pid.update(y_err, sleep=0)
            z_err = z_pid.update(z_err, sleep=0)
            yaw_err = yaw_pid.update(deg, sleep=0)

            print("errs:", x_err, y_err, z_err, yaw_err)
            
            xv = int(mss(x_err*2))
            yv = int(mss(y_err))
            zv = int(mss(z_err * 1.5 if z_err < 0 else z_err, 50))
            rv = int(mss(yaw_err, 50))
            # print(xv, yv, zv, rv)
            # drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if markId == 0: # Follow the marker
                if hasMarker(markerIds, 4):
                    return
                else:
                    zv = zv // 2 if zv < 0 else int(zv / 2.5)
                    
                    drone.send_rc_control(0, zv, yv, rv)
            elif abs(x_err) <= error[markId][0] and abs(y_err) <= error[markId][1] and abs(z_err) <= error[markId][2]:
                print("Saw marker", markId)
                return
            else: 
                drone.send_rc_control(xv, zv//2, yv, 0)
        else:
            if markId == 2: # 看不到 Marker2 的話，往下飛
                drone.send_rc_control(0, -3, 0, 0)
            else:
                drone.send_rc_control(0, 0, 0, 0)

def detect(drone, markId):
    frame = drone.get_frame_read().frame
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if markerIds is None:
        return False
    for i, id in enumerate(markerIds):
        if id[0] == markId:
            return True
    return False

def auto(drone):
    drone.takeoff()
    while not detect(drone, 1):
        drone.send_rc_control(0, 0, 50, 0)

    ## 1
    see(drone, 1)
    # drone.land()
    drone.move("right", 60)
    see(drone, 2)
    drone.move("left", 70)
    drone.move("forward", 70)

    ## 2: Go down and pass under the table.
    drone.move("down", 50)
    see(drone, 3)
    drone.move("down", 50)
    drone.move("forward", 100)

    ## 3: Follow the marker
    see(drone, 0)

    ## 4: Turn right 90 degrees and fly forward.
    see(drone, 4)
    drone.rotate_clockwise(90)

    ## 5: 
    see(drone, 5)
    drone.move("left", 300)
    drone.move("back", 100)

    ## 6: See the marker and land.
    see(drone, 6)
    # drone.move("back", 100)
    drone.land()
    
def test(drone):
    ## 5: 
    see(drone, 5)
    drone.move("left", 300)
    drone.move("back", 60)

    ## 6: See the marker and land.
    see(drone, 6)
    drone.send_rc_control(0, 0, 0, 0)
    # drone.move("back", 80)
    drone.land()

if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()

    frame_read = drone.get_frame_read()
    # calibration(frame_read)

    # teleop(drone)
    # see(drone, 1)
    # print("done")
    # auto(drone)
    test(drone)