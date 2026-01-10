import pyautogui
import mss
import numpy as np
import cv2
import time
import sys
import os
import pytesseract

# Enable Fail-Safe
pyautogui.FAILSAFE = True

def select_roi(img, window_name="Select ROI"):
    print(f"In the '{window_name}' window, drag a box around the target area.")
    print("Press ENTER to confirm your selection.")
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi

def main():
    print("=== Fit Cats Game Loop & OCR Test ===")
    print("SAFETY: Drag mouse to any corner of the screen to STOP immediately.")
    
    # --- Load Templates ---
    if not os.path.exists("template_play.png") or not os.path.exists("template_restart.png"):
        print("Error: Templates not found! Please run the capture script first.")
        return
    template_play = cv2.imread("template_play.png")
    template_restart = cv2.imread("template_restart.png")
    
    # --- CALIBRATION ---
    print("\n--- CALIBRATION ---")
    input("Press Enter when ready to calibrate Game Area...")
    
    print("Hover over the TOP-LEFT corner of the entire game area...")
    time.sleep(3)
    x1, y1 = pyautogui.position()
    print(f"Capture Top-Left: ({x1}, {y1})")
    
    print("Hover over the BOTTOM-RIGHT corner of the entire game area...")
    time.sleep(3)
    x2, y2 = pyautogui.position()
    print(f"Capture Bottom-Right: ({x2}, {y2})")
    
    monitor = {"top": y1, "left": x1, "width": x2-x1, "height": y2-y1}
    sct = mss.mss()
    
    # --- Calibrate with Screenshot ---
    print("Taking a screenshot of the game area for precise calibration...")
    game_img = np.array(sct.grab(monitor))
    game_img = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
    
    # Calibrate Click Area
    click_roi = select_roi(game_img.copy(), "Select CLICKABLE BOX (Left to Right)")
    if not click_roi: return
    min_x_rel, _, click_w, _ = click_roi
    min_x = x1 + min_x_rel
    max_x = min_x + click_w
    
    # Calibrate Score Area
    score_roi = select_roi(game_img.copy(), "Select SCORE Area")
    if not score_roi: return

    # Calibrate Agent View Area
    agent_view_roi = select_roi(game_img.copy(), "Select AGENT's VIEW (the play area)")
    if not agent_view_roi: return
    
    # Calibrate Next Cat Area
    next_cat_roi = select_roi(game_img.copy(), "Select NEXT CAT Area")
    if not next_cat_roi: return
    
    drop_y = y1 + int(monitor["height"] * 0.15)
    
    print("\n--- CALIBRATION COMPLETE ---")
    print(f"Game Region: {monitor}")
    print(f"Click X Range: {min_x} to {max_x}")
    print(f"Score ROI (relative): {score_roi}")
    print(f"Agent View ROI (relative): {agent_view_roi}")
    print(f"Next Cat ROI (relative): {next_cat_roi}")
    print("\nStarting Loop... Press Ctrl+C in the terminal to stop.")
    
    try:
        while True:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Check Restart
            res_restart = cv2.matchTemplate(img, template_restart, cv2.TM_CCOEFF_NORMED)
            _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_restart)
            
            if max_val_r > 0.8:
                print(f"State: GAME OVER. Clicking Restart...")
                cx = x1 + max_loc_r[0] + template_restart.shape[1]//2
                cy = y1 + max_loc_r[1] + template_restart.shape[0]//2
                pyautogui.click(cx, cy)
                time.sleep(2)
                continue

            # Check Play
            res_play = cv2.matchTemplate(img, template_play, cv2.TM_CCOEFF_NORMED)
            _, max_val_p, _, max_loc_p = cv2.minMaxLoc(res_play)
            
            if max_val_p > 0.8:
                print(f"State: MENU. Clicking Play...")
                cx = x1 + max_loc_p[0] + template_play.shape[1]//2
                cy = y1 + max_loc_p[1] + template_play.shape[0]//2
                pyautogui.click(cx, cy)
                time.sleep(2)
                continue
            
            # --- Playing and Reading Score ---
            sx, sy, sw, sh = score_roi
            score_img = img[sy:sy+sh, sx:sx+sw]
            
            gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            score_text = pytesseract.image_to_string(thresh, config=custom_config)
            
            try:
                score = int(score_text.strip())
            except:
                score = -1
            
            print(f"State: PLAYING. Score: {score}. Dropping cat...")
            drop_x = np.random.randint(min_x, max_x)
            pyautogui.click(drop_x, drop_y)
            time.sleep(1.0) 
            
    except pyautogui.FailSafeException:
        print("FAIL-SAFE TRIGGERED. Stopping.")
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()
