import cv2 as cv
import numpy as np
import json
import os
import time

# Force X11 to avoid the Wayland/Qt crash
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIG ---
URL = "http://192.168.53.56:81/stream"
SQUARES_X, SQUARES_Y = 10, 7
SQUARE_LENGTH, MARKER_LENGTH = 0.04, 0.03
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

board = cv.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, DICT)
detector = cv.aruco.CharucoDetector(board)

def main():
    cap = cv.VideoCapture(URL)
    if not cap.isOpened():
        print("Error: Could not connect to stream.")
        return

    # Standardize Window
    win_name = "OpenCV Calibration Suite"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    all_corners, all_ids = [], []
    imsize = None
    last_cap_time = time.time()
    calibrated = False
    mtx, dist, new_mtx = None, None, None

    print("\n--- PHASE 1: CALIBRATION ---")
    print("Move camera. Capture 15+ frames. Press [ENTER] to switch to Live Preview.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if imsize is None:
            imsize = (frame.shape[1], frame.shape[0])
            cv.resizeWindow(win_name, imsize[0] * (2 if calibrated else 1), imsize[1])

        if not calibrated:
            # --- CALIBRATION LOGIC ---
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)
            vis = frame.copy()

            if charuco_ids is not None:
                cv.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
                if time.time() - last_cap_time > 1.2 and len(charuco_ids) > 6:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    last_cap_time = time.time()
                    print(f"Captured {len(all_corners)}")
                    cv.rectangle(vis, (0,0), (imsize[0], imsize[1]), (0, 255, 0), 10)

            cv.putText(vis, f"Captured: {len(all_corners)} | [ENTER] to Finish", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.imshow(win_name, vis)
        else:
            # --- LIVE COMPARISON LOGIC ---
            undistorted = cv.undistort(frame, mtx, dist, None, new_mtx)
            comparison = np.hstack((frame, undistorted))
            
            cv.putText(comparison, "ORIGINAL", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(comparison, "UNDISTORTED", (imsize[0] + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow(win_name, comparison)

        key = cv.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 13 and not calibrated: # ENTER
            if len(all_corners) >= 10:
                print("\nCalculating math... please wait.")
                # Universal 4.10+ Method
                obj_points, img_points = [], []
                for i in range(len(all_corners)):
                    objp, imgp = board.matchImagePoints(all_corners[i], all_ids[i])
                    if objp is not None:
                        obj_points.append(objp)
                        img_points.append(imgp)
                
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, imsize, None, None)
                new_mtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, imsize, 1, imsize)
                
                # --- ADD THIS TO SAVE THE DATA ---
                calib_data = {
                    "rms_error": ret,
                    "camera_matrix": mtx.tolist(),
                    "dist_coeffs": dist.tolist(),
                    "resolution": imsize
                }
                with open("calibration.json", "w") as f:
                    json.dump(calib_data, f, indent=4)

                print("Data saved to 'calibration.json' in your current folder.")
                # ---------------------------------
                calibrated = True
                print(f"Success! RMS Error: {ret:.4f}")
                print("--- PHASE 2: LIVE COMPARISON ---")
                cv.resizeWindow(win_name, imsize[0] * 2, imsize[1])
            else:
                print("Capture more frames first!")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
