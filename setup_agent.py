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

def capture_template(prompt, output_filename, window_title):
    print(f"\n--- {window_title} ---")
    print(prompt)
    choice = input(f"Press ENTER to capture, or 'n' to skip if you have a good '{output_filename}': ").lower()
    
    if choice == 'n':
        print(f"Skipping '{output_filename}'.")
        return None

    with mss.mss() as sct:
        # In the sandbox, monitor[1] is the virtual screen
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

def main():
    print("=== Fit Cats Agent Setup Tool ===")
    print("This tool will capture templates AND calibrate coordinates.")
    
    calibration_data = {}

    # --- 0. Newgrounds Start Button ---
    ng_btn = capture_template(
        "If you see a 'Click to Play' overlay, capture it. Otherwise skip.",
        "template_newgrounds_play.png",
        "Step 0: Newgrounds Play Button"
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
        "Step 1: Select the ENTIRE GAME WINDOW"
    )
    if game_window == "abort": return
    if game_window is None:
        print("Cannot proceed without defining the game window. Aborting.")
        return
    
    calibration_data["game_width"] = game_window["w"]
    calibration_data["game_height"] = game_window["h"]

    # --- 2. Play Button ---
    play_btn = capture_template(
        "Select the PLAY button.",
        "template_play.png",
        "Step 2: Select the PLAY button"
    )
    if play_btn == "abort": return
    
    # --- 2b. Music Button ---
    music_btn = capture_template(
        "Select the MUSIC/SOUND button (to mute it).",
        "template_music.png",
        "Step 2b: Select the MUSIC button"
    )
    if music_btn == "abort": return

    # --- 3. Click Area Calibration ---
    print("\n--- Step 3: Calibrate Click Area ---")
    print("We need to define the left and right limits for dropping cats.")
    input("Press Enter to take a screenshot for calibration...")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    click_roi = select_roi(screenshot_bgr, "Select the CLICKABLE AREA (Left to Right)")
    if not click_roi: return
    
    # Calculate relative coordinates
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
        "Step 4: Select the SCORE area"
    )
    if score_roi_abs == "abort": return
    
    if score_roi_abs:
        calibration_data["score_roi"] = {
            "x": score_roi_abs["x"] - game_window["x"],
            "y": score_roi_abs["y"] - game_window["y"],
            "w": score_roi_abs["w"],
            "h": score_roi_abs["h"]
        }

    # --- 5. Next Cat Area ---
    next_cat_abs = capture_template(
        "Select the NEXT CAT box.",
        "template_next_cat_debug.png",
        "Step 5: Select the NEXT CAT area"
    )
    if next_cat_abs == "abort": return
    
    if next_cat_abs:
        calibration_data["next_cat_roi"] = {
            "x": next_cat_abs["x"] - game_window["x"],
            "y": next_cat_abs["y"] - game_window["y"],
            "w": next_cat_abs["w"],
            "h": next_cat_abs["h"]
        }
        
    # --- 6. Agent View Area ---
    agent_view_abs = capture_template(
        "Select the PLAY AREA (what the agent sees).",
        "template_empty_board.png",
        "Step 6: Select the AGENT VIEW area"
    )
    if agent_view_abs == "abort": return
    
    if agent_view_abs:
        calibration_data["agent_view_roi"] = {
            "x": agent_view_abs["x"] - game_window["x"],
            "y": agent_view_abs["y"] - game_window["y"],
            "w": agent_view_abs["w"],
            "h": agent_view_abs["h"]
        }

    # --- 7. Restart Button ---
    print("\n--- Step 7: Restart Button ---")
    print("Play until GAME OVER.")
    input("Press Enter when GAME OVER screen is visible...")
    
    restart_btn = capture_template(
        "Select the RESTART button.",
        "template_restart.png",
        "Step 7: Select the RESTART button"
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
