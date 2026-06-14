import argparse
import pyautogui
import mss
import numpy as np
import cv2
import os
import time
import json
import subprocess

def select_roi(img, window_name="Select ROI"):
    print(f"\nIn the '{window_name}' window, drag a box around the target area.")
    print("Press ENTER to confirm your selection, or 'c' to cancel.")
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi

def select_point(img, window_name="Select Point"):
    print(f"\nIn the '{window_name}' window, click a single point.")
    print("Press ENTER when done.")
    
    point = None
    
    def click_event(event, x, y, flags, params):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            img_copy = img.copy()
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, img_copy)

    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return point

def select_polygon(img, window_name="Select Polygon"):
    print(f"\nIn the '{window_name}' window, click points to define the shape.")
    print("Press ENTER when done.")
    
    points = []
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, img)

    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print("Not enough points selected.")
        return None
    return points

def capture_template(prompt, output_filename, window_title, display_id):
    print(f"\n--- {window_title} ---")
    print(prompt)
    choice = input(f"Press ENTER to capture, or 'n' to skip if you have a good '{output_filename}': ").lower()
    
    if choice == 'n':
        print(f"Skipping '{output_filename}'.")
        return None

    with mss.mss(display=display_id) as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    roi = select_roi(screenshot_bgr, window_title)
    if not roi:
        print("Selection cancelled.")
        return "abort"
        
    x, y, w, h = roi
    template_img = screenshot_bgr[y:y+h, x:x+w]
    cv2.imwrite(output_filename, template_img)
    print(f"Saved '{output_filename}'.")
    return {"x": x, "y": y, "w": w, "h": h}

def find_template_on_screen(template_filename, display_id):
    if not os.path.exists(template_filename):
        return None
    
    template = cv2.imread(template_filename)
    with mss.mss(display=display_id) as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        res = cv2.matchTemplate(screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.8:
            return {"x": max_loc[0], "y": max_loc[1], "w": template.shape[1], "h": template.shape[0]}
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=str, default=":99", help="The sandbox display ID to capture from.")
    args = parser.parse_args()
    display_id = args.sandbox

    print("=== Fit Cats Agent Setup Tool ===")
    print("This tool will capture templates AND calibrate coordinates.")
    
    calibration_data = {}

    # --- -1. Registration Popup ---
    print("\n--- Step -1: Registration Popup Handling ---")
    reg_popup = capture_template(
        "If you see a 'Register / Login' popup, capture a unique part of it. Otherwise, skip.",
        "template_register_popup.png",
        "Step -1a: Registration Popup Template",
        display_id
    )
    if reg_popup == "abort": return
    
    print("\nNow, we need to define a 'Safe Dismiss Point'.")
    print("This is a point on the screen (usually the gray area outside the popup) that is safe to click to close any overlay.")
    choice = input("Press ENTER to define this point, or 'n' to skip: ").lower()
    
    if choice != 'n':
        with mss.mss(display=display_id) as sct:
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        dismiss_point = select_point(screenshot_bgr, "Click Safe Dismiss Point")
        if dismiss_point:
            calibration_data["register_dismiss_point"] = dismiss_point
            print("Saved registration dismiss point.")
            # Use pyautogui with the correct display context
            original_display = os.environ.get("DISPLAY")
            os.environ["DISPLAY"] = display_id
            pyautogui.click(dismiss_point[0], dismiss_point[1])
            if original_display:
                os.environ["DISPLAY"] = original_display
            else:
                del os.environ["DISPLAY"]
            time.sleep(2)
    elif "register_dismiss_point" not in calibration_data and os.path.exists("calibration_data.json"):
         with open("calibration_data.json", "r") as f:
             old_calib = json.load(f)
             if "register_dismiss_point" in old_calib:
                 calibration_data["register_dismiss_point"] = old_calib["register_dismiss_point"]
                 print("Preserved existing dismiss point.")

    # --- 0. Newgrounds Start Button ---
    ng_btn = capture_template(
        "If you see a 'Click to Play' overlay, capture it. Otherwise skip.",
        "template_newgrounds_play.png",
        "Step 0: Newgrounds Play Button",
        display_id
    )
    if ng_btn == "abort": return
    
    if ng_btn:
        print("Clicking button to load game...")
        input("Please manually click the button to load the game, then press Enter...")
        time.sleep(5)

    # --- 1. Title Screen (Game Window Anchor) ---
    game_window = capture_template(
        "Make sure the game's TITLE SCREEN is visible.",
        "game_title.png",
        "Step 1: Select the ENTIRE GAME WINDOW",
        display_id
    )
    
    if game_window == "abort": return
    
    if game_window is None:
        print("Searching for existing 'game_title.png' on screen...")
        game_window = find_template_on_screen("game_title.png", display_id)
        if game_window is None:
            print("Error: Could not find 'game_title.png' on screen. You must capture it first.")
            return
        print(f"Found game window at: {game_window}")
    
    calibration_data["game_width"] = game_window["w"]
    calibration_data["game_height"] = game_window["h"]

    # --- 2. Play Button ---
    play_btn = capture_template(
        "Select the PLAY button.",
        "template_play.png",
        "Step 2: Select the PLAY button",
        display_id
    )
    if play_btn == "abort": return
    
    # --- 2b. Music Button ---
    music_btn = capture_template(
        "Select the MUSIC/SOUND button (to mute it).",
        "template_music.png",
        "Step 2b: Select the MUSIC button",
        display_id
    )
    if music_btn == "abort": return

    # --- 3. Click Area Calibration ---
    print("\n--- Step 3: Calibrate Click Area ---")
    print("Please select the FULL WIDTH of the playable area (from left wall to right wall).")
    input("Press Enter to take a screenshot for calibration...")
    
    with mss.mss(display=display_id) as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    click_roi = select_roi(screenshot_bgr, "Select the FULL CLICKABLE AREA (Left Wall to Right Wall)")
    if not click_roi: return
    
    rel_min_x = click_roi[0] - game_window["x"]
    rel_max_x = (click_roi[0] + click_roi[2]) - game_window["x"]
    
    calibration_data["click_x_min_rel"] = rel_min_x
    calibration_data["click_x_max_rel"] = rel_max_x
    
    # --- 4. Score Area ---
    print("\n--- Step 4: Score Area ---")
    print("Manually START the game so the score is visible.")
    input("Press Enter when the game is active...")
    
    score_roi_abs = capture_template(
        "Select the SCORE area.",
        "template_score_debug.png", 
        "Step 4: Select the SCORE area",
        display_id
    )
    if score_roi_abs == "abort": return
    
    if score_roi_abs:
        calibration_data["score_roi"] = {
            "x": score_roi_abs["x"] - game_window["x"],
            "y": score_roi_abs["y"] - game_window["y"],
            "w": score_roi_abs["w"],
            "h": score_roi_abs["h"]
        }
    elif "score_roi" not in calibration_data and os.path.exists("calibration_data.json"):
         with open("calibration_data.json", "r") as f:
             old_calib = json.load(f)
             if "score_roi" in old_calib:
                 calibration_data["score_roi"] = old_calib["score_roi"]
                 print("Preserved existing score ROI.")

    # --- 5. Next Cat Area ---
    next_cat_abs = capture_template(
        "Select the NEXT CAT box.",
        "template_next_cat_debug.png",
        "Step 5: Select the NEXT CAT area",
        display_id
    )
    if next_cat_abs == "abort": return
    
    if next_cat_abs:
        calibration_data["next_cat_roi"] = {
            "x": next_cat_abs["x"] - game_window["x"],
            "y": next_cat_abs["y"] - game_window["y"],
            "w": next_cat_abs["w"],
            "h": next_cat_abs["h"]
        }
    elif "next_cat_roi" not in calibration_data and os.path.exists("calibration_data.json"):
         with open("calibration_data.json", "r") as f:
             old_calib = json.load(f)
             if "next_cat_roi" in old_calib:
                 calibration_data["next_cat_roi"] = old_calib["next_cat_roi"]
                 print("Preserved existing next cat ROI.")
        
    # --- 6. Agent View Area ---
    agent_view_abs = capture_template(
        "Select the PLAY AREA (what the agent sees). Include the box and the area above it.",
        "template_empty_board.png", 
        "Step 6: Select the AGENT VIEW area",
        display_id
    )
    if agent_view_abs == "abort": return
    
    if agent_view_abs:
        calibration_data["agent_view_roi"] = {
            "x": agent_view_abs["x"] - game_window["x"],
            "y": agent_view_abs["y"] - game_window["y"],
            "w": agent_view_abs["w"],
            "h": agent_view_abs["h"]
        }
    elif "agent_view_roi" not in calibration_data and os.path.exists("calibration_data.json"):
         with open("calibration_data.json", "r") as f:
             old_calib = json.load(f)
             if "agent_view_roi" in old_calib:
                 calibration_data["agent_view_roi"] = old_calib["agent_view_roi"]
                 print("Preserved existing agent view ROI.")

    # --- 6b. Playable Area Polygon (New) ---
    print("\n--- Step 6b: Define Playable Area ---")
    print("We need to define the single polygon that covers the entire area where cats can be.")
    print("This includes the main box AND the lid area above it.")
    choice = input("Press ENTER to define points, or 'n' to skip: ").lower()
    
    if choice != 'n':
        with mss.mss(display=display_id) as sct:
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        poly_points = select_polygon(screenshot_bgr, "Select Playable Area Polygon")
        
        if poly_points:
            rel_points = []
            for (px, py) in poly_points:
                rel_points.append((px - game_window["x"], py - game_window["y"]))
            
            calibration_data["playable_polygon"] = rel_points
            print("Playable polygon saved.")
    elif "playable_polygon" not in calibration_data and os.path.exists("calibration_data.json"):
         with open("calibration_data.json", "r") as f:
             old_calib = json.load(f)
             if "playable_polygon" in old_calib:
                 calibration_data["playable_polygon"] = old_calib["playable_polygon"]
                 print("Preserved existing playable polygon.")

    # --- 7. Restart Button ---
    print("\n--- Step 7: Restart Button ---")
    print("Play until GAME OVER.")
    input("Press Enter when GAME OVER screen is visible...")
    
    restart_btn = capture_template(
        "Select the RESTART button.",
        "template_restart.png",
        "Step 7: Select the RESTART button",
        display_id
    )
    if restart_btn == "abort": return

    # Save Calibration Data
    with open("calibration_data.json", "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print("\n=== Setup Complete! ===")
    print("Templates saved.")
    print("Calibration data saved to 'calibration_data.json'.")
    print("You can now run the training script.")

if __name__ == "__main__":
    main()
