import cv2
import numpy as np
import mss
import os
import pytesseract
import json
import time
import pyautogui

def ocr_by_contour(img):
    """
    Performs OCR by finding contours, sorting them left-to-right,
    and running Tesseract on each individual digit.
    """
    # --- Corrected Pipeline ---
    # 1. Create a WHITE text on BLACK background image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use a fixed threshold, as Otsu can be unstable with single colors
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("debug_contour_thresh.png", thresh)

    # 2. Erode the WHITE text to break connections
    kernel = np.ones((3,2), np.uint8) # Use a rectangular kernel, wider than it is tall
    eroded_img = cv2.erode(thresh, kernel, iterations=1)
    cv2.imwrite("debug_contour_eroded.png", eroded_img)

    # 3. Find contours on the eroded image
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours.") # Debug print
    
    if not contours:
        return ""

    # Draw contours for debugging
    img_with_boxes = img.copy()
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort bounding boxes by their x-coordinate
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    digits = ""
    for (x, y, w, h) in sorted_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Basic filtering to ignore noise
        if w < 4 or h < 8:
            continue
            
        # Crop from the original (non-eroded) thresholded image
        padding = 3
        digit_roi = thresh[y-padding:y+h+padding, x-padding:x+w+padding]
        
        # Use PSM 10: Treat the image as a single character
        config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
        digit_text = pytesseract.image_to_string(digit_roi, config=config)
        
        digits += digit_text.strip()
        
    cv2.imwrite("debug_contour_boxes.png", img_with_boxes)
    return digits

def main():
    print("=== OCR Debugging Tool (Continuous Mode) ===")
    
    # --- Load Calibration and Templates ---
    if not os.path.exists("calibration_data.json") or not os.path.exists("game_title.png"):
        print("Error: 'calibration_data.json' or 'game_title.png' not found. Run setup_agent.py first.")
        return

    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
    game_title_template = cv2.imread("game_title.png", cv2.IMREAD_COLOR)
    
    if os.path.exists("template_newgrounds_play.png"):
        template_ng_play = cv2.imread("template_newgrounds_play.png", cv2.IMREAD_COLOR)
    else:
        template_ng_play = None

    # --- Find Game Window ---
    with mss.mss() as sct:
        print("Locating game window...")
        
        def scan_for(template):
            if template is None: return 0, (0,0), None
            full_screenshot = np.array(sct.grab(sct.monitors[0]))
            full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
            
            if full_screenshot_bgr.shape[0] < template.shape[0] or full_screenshot_bgr.shape[1] < template.shape[1]:
                return 0, (0, 0), None
                
            res = cv2.matchTemplate(full_screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            return max_val, max_loc, full_screenshot_bgr

        max_val, max_loc, _ = scan_for(game_title_template)
        
        if max_val < 0.8 and template_ng_play is not None:
            print("Title not found. Checking for Newgrounds overlay...")
            max_val_ng, max_loc_ng, _ = scan_for(template_ng_play)
            if max_val_ng > 0.8:
                print("Found Newgrounds button! Clicking it...")
                virtual_left = sct.monitors[0]['left']
                virtual_top = sct.monitors[0]['top']
                btn_x = virtual_left + max_loc_ng[0] + template_ng_play.shape[1] // 2
                btn_y = virtual_top + max_loc_ng[1] + template_ng_play.shape[0] // 2
                pyautogui.click(btn_x, btn_y)
                print("Waiting 8 seconds for game to load...")
                time.sleep(8)
                max_val, max_loc, _ = scan_for(game_title_template)

        if max_val < 0.8:
            print(f"Could not find game window. Is it visible?")
            return
            
        virtual_left = sct.monitors[0]['left']
        virtual_top = sct.monitors[0]['top']
        game_region = {
            "top": virtual_top + max_loc[1], 
            "left": virtual_left + max_loc[0], 
            "width": calib["game_width"], 
            "height": calib["game_height"]
        }
        print(f"Game window found at: {game_region}")
        
        # --- Continuous Debug Loop ---
        print("\nContinuously analyzing score. Press Ctrl+C to quit.")
        while True:
            try:
                game_img = np.array(sct.grab(game_region))
                game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
                
                roi = calib["score_roi"]
                score_img = game_img_bgr[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
                
                # --- Contour Method ---
                text_contour = ocr_by_contour(score_img)

                print(f"Contour Method: '{text_contour.strip():<5}'")

                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nExiting OCR Debugger.")
                break
            except Exception as e:
                print(f"Error during OCR: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()
