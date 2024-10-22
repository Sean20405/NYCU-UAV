import cv2
from djitellopy import Tello
import matplotlib.pyplot as plt
# Load the predefined dictionary
drone = Tello()
drone.connect()
drone.streamon()
# cap = cv2.VideoCapture(0)

f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()
f.release()

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

fig_frame, ax_frame = plt.subplots()
fig_axis, ax_axis = plt.subplots()

while True:
    # Get the frame from the drone
    frame = drone.get_frame_read()
    frame = frame.frame
    # ret, frame = cap.read()
    h, w = frame.shape[:2]
    # cv2.imwrite("output/frame.jpg", frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig_frame.suptitle('Frame')
    ax_frame.clear()  # Clear previous image
    ax_frame.imshow(frame)
    ax_frame.axis('off')  # Optional: Turn off axis
    plt.pause(0.01)

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    frame = cv2.aruco.drawDetectedMarkers(frame,markerCorners, markerIds)
    rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

    if rvec is not None:
        print("rvec: ", rvec)
        print("tvec: ", tvec)
        mark_frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 10)
        x, y, z = round(tvec[0][0][0], 2), round(tvec[0][0][1], 2), round(tvec[0][0][2], 2)
        coord = (markerCorners[0][0][0] + markerCorners[0][0][1])
        cv2.putText(mark_frame, f'x: {x}, y: {y}, z: {z}', (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite('output/drawAxis.jpg', mark_frame)
        # cv2.imshow("frame", mark_frame)
        cv2.waitKey(100)
        fig_axis.suptitle('Draw Axis')
        ax_axis.clear()  # Clear previous image
        ax_axis.imshow(mark_frame)
        ax_axis.axis('off')  # Optional: Turn off axis
        # plt.pause(0.01)
    