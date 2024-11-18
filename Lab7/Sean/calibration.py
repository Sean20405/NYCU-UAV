import cv2
import numpy as np

CALIBRATION_TIMES = 50

def calibration(cap, times=50):
    print("Calibration...")
    # cap = cv2.VideoCapture(0)
    cnt = 0
    img_pts = []

    # Read chessboard corners
    while True:
        print(cnt)
        while True:
            ret, frame = cap.read()
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
        if cnt >= times:
            break

        cv2.waitKey(33)
    
    cv2.destroyAllWindows()

    # Generate object points
    obj_pts = np.array([[[j, i, 0] for i in range(6) for j in range(9)] for _ in range(times)], dtype=np.float32)

    # Camera calibration
    ret, camera_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame.shape, None, None)

    # Save parameters
    f = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", camera_mat)
    f.write("distortion", dist_coeff)
    f.release()

if __name__ == "__main__":
    # Tello
    cap = cv2.VideoCapture(0)
    calibration(cap)
