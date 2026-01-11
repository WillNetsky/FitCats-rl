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
    Performs OCR by finding contours and testing multiple Tesseract configs on each digit.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((2,2), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "", "", ""

    min_w, min_h = 5, 10
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > min_w and cv2.boundingRect(c)[3] > min_h]
    
    if not bounding_boxes:
        return "", "", ""

    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    # --- Test different configs ---
    digits_config1 = "" # OEM 3, PSM 10 (Current)
    digits_config2 = "" # OEM 3, PSM 8
    digits_config3 = "" # OEM 0, PSM 10 (Legacy)

    for i, (x, y, w, h) in enumerate(sorted_boxes):
        padding = 5
        digit_roi = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
        digit_roi_inverted = cv2.bitwise_not(digit_roi)

        # Config 1: Modern Engine, Single Char
        config1 = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
        digit_text1 = pytesseract.image_to_string(digit_roi_inverted, config=config1)
        digits_config1 += digit_text1.strip()

        # Config 2: Modern Engine, Single Word
        config2 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        digit_text2 = pytesseract.image_to_string(digit_roi_inverted, config=config2)
        digits_config2 += digit_text2.strip()

        # Config 3: Legacy Engine, Single Char
        config3 = r'--oem 0 --psm 10 -c tessedit_char_whitelist=0123456789'
        digit_text3 = pytesseract.image_to_string(digit_roi_inverted, config=config3)
        digits_config3 += digit_text3.strip()
        
    return digits_config1, digits_config2, digits_config3

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
        print("Testing: (1) Modern/Char | (2) Modern/Word | (3) Legacy/Char")
        while True:
            try:
                game_img = np.array(sct.grab(game_region))
                game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
                
                roi = calib["score_roi"]
                score_img = game_img_bgr[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
                
                # --- Contour Method ---
                text1, text2, text3 = ocr_by_contour(score_img)

                print(f"1: '{text1.strip():<5}' | 2: '{text2.strip():<5}' | 3: '{text3.strip():<5}'")

                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nExiting OCR Debugger.")
                break
            except Exception as e:
                print(f"Error during OCR: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()
