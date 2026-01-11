import cv2
import numpy as np
import mss
import os
import json
import time
import pyautogui

def get_digit_templates(template_dir):
    """Loads all template exemplars from subdirectories."""
    digit_templates = {}
    if not os.path.exists(template_dir):
        return digit_templates
    for digit_folder in os.listdir(template_dir):
        if not digit_folder.isdigit():
            continue
        digit_path = os.path.join(template_dir, digit_folder)
        if os.path.isdir(digit_path):
            templates = []
            for template_file in os.listdir(digit_path):
                template_img = cv2.imread(os.path.join(digit_path, template_file), cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    templates.append(template_img)
            if templates:
                digit_templates[digit_folder] = templates
    return digit_templates

def find_contours(score_img):
    gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], None

    min_w, min_h = 5, 10
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > min_w and cv2.boundingRect(c)[3] > min_h]
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    return sorted_boxes, thresh

def recognize_score(score_img, digit_templates):
    sorted_boxes, thresh = find_contours(score_img)
    if not sorted_boxes:
        return "", [], None

    recognized_score = ""
    confidences = []
    for (x, y, w, h) in sorted_boxes:
        padding = 5
        digit_roi = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
        
        best_overall_score = 0.0
        best_overall_digit = "?"
        for digit_str, templates in digit_templates.items():
            best_score_for_this_digit = 0.0
            for template_img in templates:
                if digit_roi.shape[0] < template_img.shape[0] or digit_roi.shape[1] < template_img.shape[1]: continue
                res = cv2.matchTemplate(digit_roi, template_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_score_for_this_digit:
                    best_score_for_this_digit = max_val
            
            if best_score_for_this_digit > best_overall_score:
                best_overall_score = best_score_for_this_digit
                best_overall_digit = digit_str
        
        recognized_score += best_overall_digit
        confidences.append(best_overall_score)
    
    return recognized_score, confidences, (sorted_boxes, thresh)

def main():
    print("=== Multi-Exemplar Digit Template Tool ===")
    
    # --- Setup ---
    if not os.path.exists("calibration_data.json") or not os.path.exists("game_title.png"):
        print("Error: 'calibration_data.json' or 'game_title.png' not found. Run setup_agent.py first.")
        return

    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
    game_title_template = cv2.imread("game_title.png", cv2.IMREAD_COLOR)
    
    template_dir = "digit_templates"
    os.makedirs(template_dir, exist_ok=True)

    # --- Find Game Window (Robustly) ---
    with mss.mss() as sct:
        print("Locating game window...")
        
        def scan_for(template):
            if template is None: return 0, (0,0)
            full_screenshot = np.array(sct.grab(sct.monitors[0]))
            full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
            if full_screenshot_bgr.shape[0] < template.shape[0] or full_screenshot_bgr.shape[1] < template.shape[1]: return 0, (0,0)
            res = cv2.matchTemplate(full_screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            return max_val, max_loc

        max_val, max_loc = scan_for(game_title_template)
        
        if max_val < 0.8:
            template_ng_play = cv2.imread("template_newgrounds_play.png") if os.path.exists("template_newgrounds_play.png") else None
            if template_ng_play is not None:
                print("Title not found. Checking for Newgrounds overlay...")
                max_val_ng, max_loc_ng = scan_for(template_ng_play)
                if max_val_ng > 0.8:
                    print("Found Newgrounds button! Clicking it...")
                    virtual_left, virtual_top = sct.monitors[0]['left'], sct.monitors[0]['top']
                    btn_x = virtual_left + max_loc_ng[0] + template_ng_play.shape[1] // 2
                    btn_y = virtual_top + max_loc_ng[1] + template_ng_play.shape[0] // 2
                    pyautogui.click(btn_x, btn_y)
                    print("Waiting 8 seconds for game to load...")
                    time.sleep(8)
                    max_val, max_loc = scan_for(game_title_template)

        if max_val < 0.8:
            print(f"Could not find game window. Is it visible?")
            return
            
        virtual_left, virtual_top = sct.monitors[0]['left'], sct.monitors[0]['top']
        game_region = {"top": virtual_top + max_loc[1], "left": virtual_left + max_loc[0], "width": calib["game_width"], "height": calib["game_height"]}
        print(f"Game window found at: {game_region}")
        
        # --- Main Loop ---
        last_recognized_score = ""
        print("\nContinuously monitoring for score changes. Play the game.")
        
        while True:
            try:
                digit_templates = get_digit_templates(template_dir)
                
                tally_parts = []
                for i in range(10):
                    digit = str(i)
                    count = len(digit_templates.get(digit, []))
                    tally_parts.append(f"{digit}:{count}")
                print(f"\rTemplate Tally: [ {', '.join(tally_parts)} ] | Last Score: {last_recognized_score.ljust(5)}", end="")

                game_img = np.array(sct.grab(game_region))
                game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
                score_img = game_img_bgr[calib["score_roi"]['y']:calib["score_roi"]['y']+calib["score_roi"]['h'], calib["score_roi"]['x']:calib["score_roi"]['x']+calib["score_roi"]['w']]
                
                current_recognized_score, _, _ = recognize_score(score_img, digit_templates)

                if current_recognized_score and current_recognized_score != last_recognized_score:
                    print("\n" + "="*20)
                    
                    # Re-recognize to get fresh data for the loop
                    current_recognized_score, confidences, contour_data = recognize_score(score_img, digit_templates)
                    confidence_str = ", ".join([f"{c:.2f}" for c in confidences])
                    print(f"Score Changed! I see '{current_recognized_score}' (Conf: [{confidence_str}])")

                    key = input("Is this correct? (y/n): ").lower()

                    if key == 'y':
                        print("Great! Resuming monitoring...")
                        last_recognized_score = current_recognized_score
                    elif key == 'n':
                        correct_score_str = input("What is the correct score? ")
                        
                        sorted_boxes, thresh = contour_data

                        if len(correct_score_str) != len(sorted_boxes):
                            print(f"Error: Digit count mismatch! I found {len(sorted_boxes)} digits, you entered {len(correct_score_str)}.")
                        else:
                            for i, correct_digit in enumerate(correct_score_str):
                                (x, y, w, h) = sorted_boxes[i]
                                padding = 5
                                digit_roi = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]

                                digit_folder = os.path.join(template_dir, correct_digit)
                                os.makedirs(digit_folder, exist_ok=True)
                                num_existing = len(os.listdir(digit_folder))
                                template_path = os.path.join(digit_folder, f"{num_existing}.png")
                                cv2.imwrite(template_path, digit_roi)
                                print(f"Saved new exemplar for '{correct_digit}' to {template_path}")
                            
                        last_recognized_score = correct_score_str
                        print("Correction complete. Resuming monitoring...")
                    else:
                        print("Invalid input. Resuming monitoring...")
                        last_recognized_score = current_recognized_score

                time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nExiting template creator.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()
