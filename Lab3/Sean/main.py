import time
import cv2
import numpy as np
import skimage

def calibration():
    cap = cv2.VideoCapture(0)
    cnt = 0
    img_pts = []

    # Read chessboard corners
    while True:
        while True:
            ret, frame = cap.read()
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
        if cnt >= 4:
            break

        cv2.waitKey(1000)

    # Generate object points
    obj_pts = np.array([[[j, i, 0] for i in range(6) for j in range(9)] for _ in range(4)], dtype=np.float32)

    # Camera calibration
    ret, camera_mat, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame.shape, None, None)
    # print(camera_mat)
    # print(dist_coeff)

    # Save parameters
    f = cv2.FileStorage("param.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", camera_mat)
    f.write("distortion", dist_coeff)
    f.release()

def bilinearInterpolation(img, pt):
    h, w, _ = img.shape
    y, x, _ = pt
    x1 = np.floor(x).astype(int); y1 = np.floor(y).astype(int)
    x2 = x1 + 1                 ; y2 = y1 + 1
    
    if x1 < 0 or x2 >= h-1 or y1 < 0 or y2 >= w-1:
        return np.array([0, 0, 0])

    R1 = (x2 - x) / (x2 - x1) * img[x1, y1] + (x - x1) / (x2 - x1) * img[x2, y1]
    R2 = (x2 - x) / (x2 - x1) * img[x1, y2] + (x - x1) / (x2 - x1) * img[x2, y2]
    P = (y2 - y) / (y2 - y1) * R1 + (y - y1) / (y2 - y1) * R2
    return P

def warpToScreen():
    # screen_img = cv2.imread('screen.jpg')
    screen_img = cv2.imread('screen_10%.jpg')
    h, w, _ = screen_img.shape

    # Get perspective transform matrix
    # screen_pts = [(415, 868), (1632, 221), (1647, 1253), (333, 1408)]
    # screen_pts = [(208, 434), (816, 111), (823, 626), (167, 704)]
    # screen_pts = [(104, 217), (408, 56), (412, 313), (83, 352)]
    # screen_pts = [(83, 174), (326, 44), (330, 251), (67, 282)]
    screen_pts = [(42, 87), (163, 22), (165, 125), (33, 141)]
    frame_pts = [(0, 0), (640, 0), (640, 480), (0, 480)]
    trans_mat = cv2.getPerspectiveTransform(np.array(screen_pts, dtype=np.float32), np.array(frame_pts, dtype=np.float32))

    # Create mask to be warped
    polygon = np.array(screen_pts, np.int32)
    mask = skimage.draw.polygon2mask((w, h), polygon)
    select_pts_in_mask = np.append(np.argwhere(mask == 1), np.ones((np.count_nonzero(mask), 1)), axis=1).astype(int)

    cap = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        ret, frame = cap.read()

        # Warp frame to screen
        frame_pts = (trans_mat @ select_pts_in_mask.T).T
        frame_pts = frame_pts / frame_pts[:, 2].reshape(-1, 1)
        for i, select_pt in enumerate(select_pts_in_mask):
            screen_img[int(select_pt[1]), int(select_pt[0])] = bilinearInterpolation(frame, frame_pts[i])
        
        print("FPS:", 1 / (time.time() - start_time))

        # cv2.imshow('screen', cv2.resize(screen_img, (0, 0), fx=0.5, fy=0.5))
        # cv2.imshow('screen', screen_img)
        cv2.imshow('screen', cv2.resize(screen_img, (0, 0), fx=4, fy=4))
        cv2.waitKey(10)



if __name__ == '__main__':
    # calibration()
    warpToScreen()