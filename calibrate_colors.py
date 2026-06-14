import cv2
import numpy as np
import mss
import os
import json
import time
import argparse
from utils import start_game

# --- Cat Detection Logic ---
def find_cats_in_view(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, mask)
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 1.5)
    
    circles_big = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                   param1=50, param2=45, minRadius=40, maxRadius=100)
    circles_small = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                                     param1=50, param2=25, minRadius=15, maxRadius=40)
    
    all_circles = []
    if circles_big is not None: all_circles.extend(circles_big[0, :])
    if circles_small is not None: all_circles.extend(circles_small[0, :])
    if not all_circles: return []

    all_circles = np.uint16(np.around(all_circles))
    all_circles = sorted(all_circles, key=lambda c: c[2], reverse=True)
    
    valid_cats = []
    for i in range(len(all_circles)):
        c_curr = all_circles[i]
        is_nested = False
        for c_valid in valid_cats:
            dist = np.linalg.norm(c_curr[:2] - c_valid[:2])
            if dist < c_valid[2] and c_curr[2] < c_valid[2]:
                is_nested = True
                break
        if not is_nested:
            valid_cats.append(c_curr)
            
    return valid_cats

def get_binned_radii_auto(radii, num_clusters):
    """Automatically finds the best bins using a simple clustering method."""
    if not radii or num_clusters <= 0:
        return [], 0
    
    sorted_radii = sorted(list(set(radii)))
    if len(sorted_radii) <= num_clusters:
        return sorted_radii, 3 # Default tolerance

    # Find the largest gaps between sorted radii
    gaps = np.diff(sorted_radii)
    
    # The cluster boundaries are the indices of the largest gaps
    # We need num_clusters - 1 boundaries
    if len(gaps) < num_clusters - 1:
        # Not enough data to form the requested number of clusters
        return [], 0

    boundary_indices = np.argsort(gaps)[- (num_clusters - 1):]
    
    clusters = []
    last_idx = 0
    for boundary in sorted(boundary_indices):
        cluster = sorted_radii[last_idx : boundary + 1]
        clusters.append(cluster)
        last_idx = boundary + 1
    clusters.append(sorted_radii[last_idx:])
    
    # The representative radius for each bin is the mean of the cluster
    bin_representatives = [int(np.mean(c)) for c in clusters]
    
    # Estimate a good tolerance
    # A good tolerance is half the smallest gap between the means of the bins
    if len(bin_representatives) > 1:
        min_gap = np.min(np.diff(sorted(bin_representatives)))
        tolerance = max(1, int(min_gap / 2))
    else:
        tolerance = 3 # Default if only one bin

    return bin_representatives, tolerance

# --- Main Calibration Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=str, default=":99", help="The sandbox display ID")
    args = parser.parse_args()
    
    print("=== Automatic Cat Size Binning Tool ===")
    
    os.environ["DISPLAY"] = args.sandbox
    import pyautogui
    
    if not os.path.exists("calibration_data.json"):
        print("Error: calibration_data.json not found. Run setup_agent.py first.")
        return
    with open("calibration_data.json", "r") as f:
        calib = json.load(f)

    cv2.namedWindow("Live Agent View")
    
    detected_radii = set()
    
    with mss.mss() as sct:
        game_region = start_game(sct, calib, pyautogui)
        
        print("\nPhase 1: Automatic Detection")
        print("Play the game for ~60 seconds to expose all cat types.")
        print("The tool will automatically record all detected cat sizes.")
        print("Press 'c' to continue to the next phase.")

        start_time = time.time()
        while time.time() - start_time < 60:
            game_img = np.array(sct.grab(game_region))
            game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
            
            agent_roi = calib["agent_view_roi"]
            agent_view_img = game_img_bgr[agent_roi['y']:agent_roi['y']+agent_roi['h'], agent_roi['x']:agent_roi['x']+agent_roi['w']]
            
            mask = np.zeros((agent_view_img.shape[0], agent_view_img.shape[1]), dtype=np.uint8)
            if "playable_polygon" in calib:
                poly_points = calib["playable_polygon"]
                rel_poly_points = [(px - agent_roi['x'], py - agent_roi['y']) for px, py in poly_points]
                pts = np.array(rel_poly_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            
            detected_cats = find_cats_in_view(agent_view_img, mask)
            
            for cat in detected_cats:
                detected_radii.add(cat[2])

            cv2.imshow("Live Agent View", agent_view_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break
            
            time.sleep(0.1)

        print(f"\nDetected {len(detected_radii)} unique radii.")
        
        # --- Phase 2: Bin Calibration ---
        print("\nPhase 2: Bin Calibration")
        input("Fill the box with as many different cat types as possible, then press Enter...")
        
        while True:
            try:
                num_types_str = input("How many unique cat types are on the screen? ")
                if num_types_str.strip() == "":
                    print("Please enter a number.")
                    continue
                actual_num_types = int(num_types_str)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

        # --- KEY FIX: Use automatic clustering ---
        final_bins, best_tolerance = get_binned_radii_auto(list(detected_radii), actual_num_types)
        
        if not final_bins:
            print("Could not determine bins automatically. Please try again with more data.")
            return
            
        print(f"Automatically determined bins: {final_bins} with tolerance {best_tolerance}")
        
        # --- Save Logic ---
        print("\n--- Saving Cat Size Bins ---")
        cat_types = {}
        for i, bin_radius in enumerate(sorted(final_bins)):
            cat_types[str(i)] = {
                "radius": int(bin_radius),
                "tolerance": best_tolerance
            }
        
        calib["cat_types"] = cat_types
        print("Saved Cat Types:")
        print(json.dumps(calib["cat_types"], indent=4))
        
        with open("calibration_data.json", "w") as f:
            json.dump(calib, f, indent=4)
        print("Calibration data saved!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
