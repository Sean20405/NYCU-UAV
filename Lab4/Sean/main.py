import cv2
import numpy as np
from djitellopy import Tello

def calibration(frame_read):
    print("Calibration...")
    # cap = cv2.VideoCapture(0)
    cnt = 0
    img_pts = []

    # Read chessboard corners
    while True:
        print(cnt)
        while True:
            frame = frame_read.frame
            # ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corner = cv2.findChessboardCorners(frame, (9, 6), None)
            if ret:
                break

        cv2.cornerSubPix(
            frame, 
            corner, 
            winSize=(11, 11), 
            zeroZone=(-1, -1), 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        img_pts.append(corner)
        cnt += 1
        if cnt >= 10:
            break

        cv2.waitKey(33)
    
    cv2.destroyAllWindows()

    # Generate object points
    obj_pts = np.array([[[j, i, 0] for i in range(6) for j in range(9)] for _ in range(10)], dtype=np.float32)

    # Camera calibration
    ret, camera_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame.shape, None, None)
    # print(camera_mat)
    # print(dist_coeff)

    # Save parameters
    f = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", camera_mat)
    f.write("distortion", dist_coeff)
    f.release()

def detection(frame_read):   
    print("Detection...")
    cap = cv2.VideoCapture(0)
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Get param of camera calibration
    fs = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    while True:
        # frame = frame_read.frame
        ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if markerIds is None:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)
            continue
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # Pose estimation for single markers. 
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.15, intrinsic, distortion)
        # rmat, _ = cv2.Rodrigues(rvec)
        print(rvec)
        x, y, z = tvec[0][0]
        text_coor = (np.sum(markerCorners[0][0], axis=0) / 4).tolist()
        # print(text_coor)
        text_coor = tuple([int(i+20) for i in text_coor])
        # print(x, y, z)
        # print(text_coor)
        # print(markerCorners[0][0])
        # cv2.putText(frame, text=f'x: {round(x, 2)}  y: {round(y, 2)}  z: {round(z, 2)}', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, org=text_coor, color=(0, 255, 255), thickness=1)
        # frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 7)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)
    

if __name__ == "__main__":
    # Tello
    # drone = Tello()
    # drone.connect()
    #time.sleep(10)
    # drone.streamon()
    # frame_read = drone.get_frame_read()
    # calibration(frame_read)
    # detection(frame_read)
    # calibration(1)
    detection(1)
