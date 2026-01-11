import cv2
import numpy as np
import mss
import os
import time

# --- Copy of Calibration Data from fit_cats_env.py ---
NEXT_CAT_ROI = {'x': 1080, 'y': 60, 'w': 121, 'h': 103}

def main():
    print("=== Debugging Next Cat Size ===")
    
    if not os.path.exists("game_title.png"):
        print("Error: 'game_title.png' not found.")
        return

    game_title_template = cv2.imread("game_title.png", cv2.IMREAD_COLOR)
    
    with mss.mss() as sct:
        # 1. Locate Game Window (User must be on Title Screen)
        print("\nStep 1: Please go to the game's TITLE SCREEN.")
        input("Press Enter when the title screen is visible...")
        
        print("Locating game window...")
        full_screenshot = np.array(sct.grab(sct.monitors[0]))
        full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
        
        res = cv2.matchTemplate(full_screenshot_bgr, game_title_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val < 0.8:
            print(f"Could not find game window. Max confidence: {max_val:.2f}")
            return
            
        print(f"Game window found with confidence: {max_val:.2f}")
        
        # Calculate Game Region
        virtual_left = sct.monitors[0]['left']
        virtual_top = sct.monitors[0]['top']
        game_left = virtual_left + max_loc[0]
        game_top = virtual_top + max_loc[1]
        
        # 2. Wait for User to Start Game
        print("\nStep 2: Now START the game so the 'Next Cat' box is visible.")
        input("Press Enter when the game is active and the next cat is shown...")
        
        # 3. Capture Next Cat Area using remembered coordinates
        print("Capturing Next Cat area...")
        # Re-grab the screen
        full_screenshot = np.array(sct.grab(sct.monitors[0]))
        full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
        
        nx = game_left + NEXT_CAT_ROI['x']
        ny = game_top + NEXT_CAT_ROI['y']
        nw = NEXT_CAT_ROI['w']
        nh = NEXT_CAT_ROI['h']
        
        # Adjust for virtual screen offset when slicing
        slice_y = max_loc[1] + NEXT_CAT_ROI['y']
        slice_x = max_loc[0] + NEXT_CAT_ROI['x']
        
        next_cat_img = full_screenshot_bgr[slice_y : slice_y+nh, slice_x : slice_x+nw]
        
        cv2.imwrite("debug_next_cat_raw.png", next_cat_img)
        print("Saved 'debug_next_cat_raw.png'")
        
        # 4. Process
        gray = cv2.cvtColor(next_cat_img, cv2.COLOR_BGR2GRAY)
        # Try different thresholds
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite("debug_next_cat_thresh.png", thresh)
        print("Saved 'debug_next_cat_thresh.png'")
        
        pixel_count = np.count_nonzero(thresh)
        normalized_size = pixel_count / (nw * nh)
        
        print(f"Pixel Count: {pixel_count}")
        print(f"Total Pixels: {nw * nh}")
        print(f"Normalized Size: {normalized_size:.4f}")

if __name__ == "__main__":
    main()
