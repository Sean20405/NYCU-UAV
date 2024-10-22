from djitellopy import Tello
from getch import getch
from pyimagesearch.pid import PID
import cv2
import numpy as np

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


def teleop(drone):
    frame_read = drone.get_frame_read()
    while True:
        frame = frame_read.frame
        cv2.imshow('frame', frame)
        # key = getch()
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)

def auto(drone):
    frame_read = drone.get_frame_read()

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("param-drone.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()

    while True:
        frame = frame_read.frame
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        elif markerIds is not None:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            (x_err, y_err, z_err) = tvec[0][0]
            z_err = z_err - 1.5
            x_err = x_err * 2
            y_err = y_err * 2
            xv = x_pid.update(x_err, sleep=0)
            yv = y_pid.update(y_err, sleep=0)
            zv = z_pid.update(z_err, sleep=0)
            rv = np.array(cv2.Rodrigues(rvec)[0]).dot(np.array([0, 0, 1]))[0]*100
            print(xv, yv, zv, rv)
            drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)


if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()
    # teleop(drone)
    auto(drone)