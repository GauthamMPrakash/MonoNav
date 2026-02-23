import cv2 as cv
import numpy as np
import json

def calculate_roi_and_refined():
    # 1. Load the original calibration
    try:
        with open("calibration.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: calibration.json not found! Run the calibration script first.")
        return

    # Convert lists back to Numpy arrays
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    w, h = data["resolution"]

    # 2. Calculate the Refined Matrix and the ROI
    # alpha=0: Zoom in so all black 'undistortion' pixels are cropped out (Cleanest)
    # alpha=1: Keep all pixels, resulting in black curved edges at the corners
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0, newImgSize=(w, h))

    # 3. Print Results
    x, y, rw, rh = roi
    
    print("--- ROI ANALYSIS ---")
    print(f"Original Resolution: {w}x{h}")
    print(f"Calculated ROI: x={x}, y={y}, width={rw}, height={rh}")
    
    if rw < w or rh < h:
        lost_w = w - rw
        lost_h = h - rh
        print(f"Note: To remove distortion, you lose {lost_w}px width and {lost_h}px height.")
    else:
        print("Note: No cropping required (your lens is already very straight).")

    print("\n--- REFINED INTRINSIC MATRIX ---")
    print(new_mtx)

    # 4. Save these refined values to a new file
    refined_payload = {
        "refined_matrix": new_mtx.tolist(),
        "roi": list(roi),
        "explanation": "Use this matrix and ROI for measurements on undistorted images."
    }

    with open("refined_data.json", "w") as f:
        json.dump(refined_payload, f, indent=4)
    
    print("\nSuccess! Results saved to 'refined_data.json'")

if __name__ == "__main__":
    calculate_roi_and_refined()
