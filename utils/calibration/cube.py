import cv2 as cv
import numpy as np
import json
import os

# Fix for Linux Wayland/Qt error
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIG ---
URL = "http://192.168.53.56:81/stream"
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
SQUARES_X, SQUARES_Y = 10, 7
SQUARE_SIZE = 0.04 # 4cm in meters
MARKER_SIZE = 0.03 # 3cm in meters

board = cv.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, DICT)
detector = cv.aruco.CharucoDetector(board)

def main():
    # 1. Load Calibration
    if not os.path.exists("calibration.json"):
        print("Error: Run calibration script first!")
        return
    with open("calibration.json", "r") as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])

    cap = cv.VideoCapture(URL)
    
    # --- STICKY WINDOW INITIALIZATION ---
    WIN_NAME = "Sticky AR Viewer"
    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL) # Allows manual resizing
    cv.moveWindow(WIN_NAME, 100, 100) # Forces window to start at (100, 100)
    
    print("Sticky window active. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)

        if charuco_ids is not None and len(charuco_ids) > 6:
            objp, imgp = board.matchImagePoints(charuco_corners, charuco_ids)
            success, rvec, tvec = cv.solvePnP(objp, imgp, mtx, dist)

            if success:
                # --- DISTANCE CALCULATION ---
                # tvec is [x, y, z]. z is the depth (distance) in meters.
                distance_m = np.linalg.norm(tvec)
                distance_cm = distance_m * 100
                
                # --- DRAW 3D CUBE ---
                s = SQUARE_SIZE
                axis = np.float32([[0,0,0], [s,0,0], [s,s,0], [0,s,0],
                                   [0,0,-s],[s,0,-s],[s,s,-s],[0,s,-s]])
                imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                cv.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2) # Base
                for i, j in zip(range(4), range(4, 8)):
                    cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2) # Pillars
                cv.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2) # Top

                # Display Distance
                cv.putText(frame, f"Dist: {distance_cm:.1f} cm", (20, 40), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv.imshow(WIN_NAME, frame)
        if cv.waitKey(1) == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
