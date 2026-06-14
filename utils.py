import cv2
import numpy as np
import mss
import time
import os

def find_game_window(sct, game_title_template):
    """Scans the sandbox monitor for the game title template."""
    if len(sct.monitors) < 2:
        raise Exception("Sandbox monitor not found. Ensure Xephyr is running.")
        
    monitor = sct.monitors[1]
    
    screenshot = np.array(sct.grab(monitor))
    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    if screenshot.shape[0] < game_title_template.shape[0] or screenshot.shape[1] < game_title_template.shape[1]:
        return None
        
    res = cv2.matchTemplate(screenshot_bgr, game_title_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val > 0.8:
        return max_loc
    return None

def start_game(sct, calib, pyautogui):
    """
    Finds the game window, handles pre-game menus, and starts the game.
    Returns the game_region dictionary.
    """
    print("--- Starting Game ---")
    
    game_title_template = cv2.imread("game_title.png")
    ng_template = cv2.imread("template_newgrounds_play.png") if os.path.exists("template_newgrounds_play.png") else None
    play_template = cv2.imread("template_play.png")
    music_template = cv2.imread("template_music.png") if os.path.exists("template_music.png") else None
    reg_popup_template = cv2.imread("template_register_popup.png") if os.path.exists("template_register_popup.png") else None
    
    start_time = time.time()
    while time.time() - start_time < 60:
        if len(sct.monitors) < 2:
            raise Exception("Sandbox monitor not found.")
            
        monitor = sct.monitors[1]
        full_screenshot = np.array(sct.grab(monitor))
        full_screenshot_bgr = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)

        # 1. Handle Registration Popup (Highest Priority)
        if reg_popup_template is not None and "register_dismiss_point" in calib:
            res_reg = cv2.matchTemplate(full_screenshot_bgr, reg_popup_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_reg, _, _ = cv2.minMaxLoc(res_reg)
            if max_val_reg > 0.8:
                print("Found Registration Popup, dismissing...")
                # --- KEY FIX: Coordinates relative to sandbox (0,0) ---
                dismiss_x = calib["register_dismiss_point"][0]
                dismiss_y = calib["register_dismiss_point"][1]
                pyautogui.click(dismiss_x, dismiss_y)
                time.sleep(2)
                continue

        # 2. Handle Newgrounds Play Button
        if ng_template is not None:
            res_ng = cv2.matchTemplate(full_screenshot_bgr, ng_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_ng, _, max_loc_ng = cv2.minMaxLoc(res_ng)
            if max_val_ng > 0.8:
                print("Found Newgrounds button, clicking...")
                # --- KEY FIX: Coordinates relative to sandbox (0,0) ---
                btn_x = max_loc_ng[0] + ng_template.shape[1] // 2
                btn_y = max_loc_ng[1] + ng_template.shape[0] // 2
                pyautogui.click(btn_x, btn_y)
                
                print("Waiting for game to load...")
                wait_start_time = time.time()
                while time.time() - wait_start_time < 20: 
                    if find_game_window(sct, game_title_template) is not None:
                        print("Game loaded!")
                        break 
                    time.sleep(0.5)
                else: 
                    print("Warning: Timed out waiting for game title after clicking Newgrounds button.")
                
                continue 

        # 3. Find the main game window by its title
        title_loc = find_game_window(sct, game_title_template)
        if title_loc:
            # --- KEY FIX: Coordinates relative to sandbox (0,0) ---
            game_region = {
                "top": title_loc[1], 
                "left": title_loc[0],
                "width": calib["game_width"], 
                "height": calib["game_height"]
            }
            print(f"Game window found at: {game_region}")
            
            # Now that we have the region, check for menu buttons
            game_img = np.array(sct.grab(game_region))
            game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
            
            # Click Music button first, if it exists and is found
            if music_template is not None:
                res_music = cv2.matchTemplate(game_img_bgr, music_template, cv2.TM_CCOEFF_NORMED)
                _, max_val_music, _, max_loc_music = cv2.minMaxLoc(res_music)
                if max_val_music > 0.8:
                    print("Found music button, clicking...")
                    btn_x = game_region['left'] + max_loc_music[0] + music_template.shape[1] // 2
                    btn_y = game_region['top'] + max_loc_music[1] + music_template.shape[0] // 2
                    pyautogui.click(btn_x, btn_y)
                    time.sleep(1) 
                    game_img = np.array(sct.grab(game_region))
                    game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)

            # Then, click the Play button
            res_play = cv2.matchTemplate(game_img_bgr, play_template, cv2.TM_CCOEFF_NORMED)
            _, max_val_play, _, max_loc_play = cv2.minMaxLoc(res_play)
            if max_val_play > 0.8:
                print("Found main menu 'Play' button, clicking...")
                btn_x = game_region['left'] + max_loc_play[0] + play_template.shape[1] // 2
                btn_y = game_region['top'] + max_loc_play[1] + play_template.shape[0] // 2
                pyautogui.click(btn_x, btn_y)
                time.sleep(2) 
            
            return game_region

        print("Waiting for game window...")
        time.sleep(0.5)
        
    raise Exception("Could not find and start the game after 60 seconds.")
