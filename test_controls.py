import cv2
import numpy as np
import mss
import os
import json
import time
import pyautogui

def find_game_window(sct, game_title_template, ng_template):
    """Finds the game window, handling the Newgrounds overlay."""
    print("Searching for game window...")
    start_time = time.time()
    while time.time() - start_time < 60:
        full_screenshot = np.array(sct.grab(sct.monitors[0]))
        full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)

        res = cv2.matchTemplate(full_screenshot_bgr, game_title_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            return max_loc

        if ng_template is not None:
            res_ng = cv2.matchTemplate(full_screenshot_bgr, ng_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_ng, _, max_loc_ng = cv2.minMaxLoc(res_ng)
            if max_val_ng > 0.8:
                print("Found Newgrounds button, clicking it...")
                btn_x = sct.monitors[0]['left'] + max_loc_ng[0] + ng_template.shape[1] // 2
                btn_y = sct.monitors[0]['top'] + max_loc_ng[1] + ng_template.shape[0] // 2
                pyautogui.click(btn_x, btn_y)
                time.sleep(8)
                continue
        
        print("Waiting for game window...")
        time.sleep(2)
    
    raise Exception("Could not find game window after 60 seconds.")

def main():
    print("=== Interactive Game Control Speed Test ===")
    
    if not os.path.exists("calibration_data.json"):
        print("Error: calibration_data.json not found. Run setup_agent.py first.")
        return
    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
    
    game_title_template = cv2.imread("game_title.png")
    play_template = cv2.imread("template_play.png")
    restart_template = cv2.imread("template_restart.png")
    ng_template = cv2.imread("template_newgrounds_play.png") if os.path.exists("template_newgrounds_play.png") else None

    with mss.mss() as sct:
        title_loc = find_game_window(sct, game_title_template, ng_template)
        game_region = {"top": title_loc[1], "left": title_loc[0], "width": calib["game_width"], "height": calib["game_height"]}
        print(f"Game window found at: {game_region}")

        # --- Binary Search for Optimal Delay ---
        low_delay = 0.05
        high_delay = 0.5
        best_working_delay = high_delay
        iterations = 5

        print("\nStarting binary search for optimal click delay...")
        print(f"Range: [{low_delay}s, {high_delay}s]")

        for i in range(iterations):
            print("\n" + "-"*30)
            print(f"Iteration {i+1}/{iterations}")
            
            game_img = np.array(sct.grab(game_region))
            game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
            
            res_play = cv2.matchTemplate(game_img_bgr, play_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_play, _, max_loc_play = cv2.minMaxLoc(res_play)

            res_restart = cv2.matchTemplate(game_img_bgr, restart_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_restart, _, max_loc_restart = cv2.minMaxLoc(res_restart)
            
            if max_val_play > 0.8:
                pyautogui.click(game_region['left'] + max_loc_play[0] + play_template.shape[1] // 2, game_region['top'] + max_loc_play[1] + play_template.shape[0] // 2)
            elif max_val_restart > 0.8:
                 pyautogui.click(game_region['left'] + max_loc_restart[0] + restart_template.shape[1] // 2, game_region['top'] + max_loc_restart[1] + restart_template.shape[0] // 2)
            
            print("Starting new game...")
            time.sleep(3)

            test_delay = (low_delay + high_delay) / 2.0
            num_clicks = 10
            print(f"Testing delay: {test_delay:.4f}s")

            min_x, max_x = calib["click_x_min_rel"], calib["click_x_max_rel"]
            center_x_rel = (min_x + max_x) // 2
            click_x_abs = game_region['left'] + center_x_rel
            drop_y_abs = game_region['top'] + int(game_region['height'] * 0.15)

            for _ in range(num_clicks):
                pyautogui.click(click_x_abs, drop_y_abs)
                time.sleep(test_delay)
            
            time.sleep(2)

            # --- Get User Confirmation ---
            try:
                answer = input("Visually inspect the game. How many cats were dropped? ")
                num_cats_found = int(answer)
            except (ValueError, TypeError):
                print("Invalid input. Assuming failure.")
                num_cats_found = 0

            if num_cats_found >= num_clicks:
                print("SUCCESS: Delay is reliable. Trying faster...")
                best_working_delay = test_delay
                high_delay = test_delay
            else:
                print("FAILURE: Delay is too fast. Trying slower...")
                low_delay = test_delay
        
        print("\n" + "="*30)
        print("Binary search complete!")
        print(f"Fastest reliable delay found: {best_working_delay:.4f}s")
        print(f"Recommended value (with safety margin): {best_working_delay * 1.2:.4f}s")

if __name__ == "__main__":
    main()
