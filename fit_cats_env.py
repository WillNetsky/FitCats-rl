import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import mss
import time
import os
import pytesseract
import json
from collections import deque

class FitCatsEnv(gym.Env):
    def __init__(self):
        super(FitCatsEnv, self).__init__()
        
        # Pyautogui and mss will now respect the DISPLAY variable
        # set by the parent script (train_distributed.py or launch_instance.sh)
        pyautogui.FAILSAFE = True

        # Load Calibration Data
        if not os.path.exists("calibration_data.json"):
            raise FileNotFoundError("calibration_data.json not found! Run setup_agent.py first.")
            
        with open("calibration_data.json", "r") as f:
            self.calib = json.load(f)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "next_cat_size": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "time_since_click": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.template_play = cv2.imread("template_play.png", cv2.IMREAD_COLOR)
        self.template_restart = cv2.imread("template_restart.png", cv2.IMREAD_COLOR)
        self.template_empty_board = cv2.imread("template_empty_board.png", cv2.IMREAD_COLOR)
        self.game_title_template = cv2.imread("game_title.png", cv2.IMREAD_COLOR)
        
        if os.path.exists("template_newgrounds_play.png"):
            self.template_ng_play = cv2.imread("template_newgrounds_play.png", cv2.IMREAD_COLOR)
        else:
            self.template_ng_play = None
            
        if os.path.exists("template_music.png"):
            self.template_music = cv2.imread("template_music.png", cv2.IMREAD_COLOR)
        else:
            self.template_music = None

        if any(t is None for t in [self.template_play, self.template_restart, self.template_empty_board, self.game_title_template]):
            raise FileNotFoundError("Required template images not found! Run setup_agent.py.")

        self.sct = mss.mss()
        self.game_region = self._locate_game_window()
        
        self.last_score = 0
        self.session_high_score = 0
        self.step_count = 0
        self.last_click_time = time.time()
        self.consecutive_waits = 0
        self.score_buffer = deque(maxlen=5)
        self.low_score_counter = 0
        self.music_muted = False

    def _locate_game_window(self):
        display = os.environ.get('DISPLAY', ':0')
        print(f"[{display}] Searching for game window...")
        
        def scan_for(template):
            full_screenshot = np.array(self.sct.grab(self.sct.monitors[0]))
            full_screenshot = cv2.cvtColor(full_screenshot, cv2.COLOR_BGRA2BGR)
            
            if full_screenshot.shape[0] < template.shape[0] or full_screenshot.shape[1] < template.shape[1]:
                return 0, (0, 0)
                
            res = cv2.matchTemplate(full_screenshot, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            return max_val, max_loc

        start_time = time.time()
        while time.time() - start_time < 60:
            max_val, max_loc = scan_for(self.game_title_template)
            if max_val > 0.8:
                return self._calculate_region(max_loc)

            if self.template_ng_play is not None:
                max_val_ng, max_loc_ng = scan_for(self.template_ng_play)
                
                if max_val_ng > 0.8:
                    print(f"[{display}] Found Newgrounds button! Clicking it...")
                    virtual_left = self.sct.monitors[0]['left']
                    virtual_top = self.sct.monitors[0]['top']
                    
                    btn_x = virtual_left + max_loc_ng[0] + self.template_ng_play.shape[1] // 2
                    btn_y = virtual_top + max_loc_ng[1] + self.template_ng_play.shape[0] // 2
                    
                    pyautogui.click(btn_x, btn_y)
                    print(f"[{display}] Waiting 8 seconds for game to load...")
                    time.sleep(8)
                    continue

            time.sleep(2)
            print(f"[{display}] Waiting for game window...")

        raise Exception(f"[{display}] Could not find the game window (Title Screen) after 60 seconds.")

    def _calculate_region(self, max_loc):
        display = os.environ.get('DISPLAY', ':0')
        virtual_left = self.sct.monitors[0]['left']
        virtual_top = self.sct.monitors[0]['top']
        
        screen_left = virtual_left + max_loc[0]
        screen_top = virtual_top + max_loc[1]
        
        region = {
            "top": screen_top, 
            "left": screen_left, 
            "width": self.calib["game_width"], 
            "height": self.calib["game_height"]
        }
        print(f"[{display}] Game window found at: {region}")
        return region

    def _find_template(self, img, template):
        if template is None or img.shape[0] < template.shape[0] or img.shape[1] < template.shape[1]:
            return 0, (0, 0)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    def _click_template(self, max_loc, template):
        cx = self.game_region['left'] + max_loc[0] + template.shape[1] // 2
        cy = self.game_region['top'] + max_loc[1] + template.shape[0] // 2
        pyautogui.click(cx, cy)

    def _read_score(self, img):
        try:
            roi = self.calib["score_roi"]
            score_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
            
            gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            score_text = pytesseract.image_to_string(thresh, config=custom_config)
            return int(score_text.strip())
        except:
            return -1

    def _get_observation(self):
        img = np.array(self.sct.grab(self.game_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Board View
        roi = self.calib["agent_view_roi"]
        board_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
        board_obs = cv2.resize(board_img, (84, 84))

        # Next Cat Size
        roi = self.calib["next_cat_roi"]
        next_cat_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
        gray = cv2.cvtColor(next_cat_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        pixel_count = np.count_nonzero(thresh)
        normalized_size = pixel_count / (roi['w'] * roi['h'])
        
        time_delta = time.time() - self.last_click_time

        return {
            "board": board_obs, 
            "next_cat_size": np.array([normalized_size], dtype=np.float32),
            "time_since_click": np.array([time_delta], dtype=np.float32)
        }

    def step(self, action):
        self.step_count += 1
        
        x_pos_norm, click_trigger = action
        did_click = click_trigger > 0
        action_str = "WAIT"

        if did_click:
            action_str = f"CLICK @ {x_pos_norm:.2f}"
            norm_action = (x_pos_norm + 1) / 2.0
            
            min_x = self.calib["click_x_min_rel"]
            max_x = self.calib["click_x_max_rel"]
            
            click_x_rel = int(norm_action * (max_x - min_x) + min_x)
            click_x_abs = self.game_region['left'] + click_x_rel
            
            # Drop Y is 15% down from top
            drop_y_abs = self.game_region['top'] + int(self.game_region['height'] * 0.15)
            
            pyautogui.click(click_x_abs, drop_y_abs)
            self.last_click_time = time.time()
            self.consecutive_waits = 0
            time.sleep(0.75)
        else:
            self.consecutive_waits += 1
            time.sleep(0.1)

        full_img = np.array(self.sct.grab(self.game_region))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)
        obs = self._get_observation()

        cat_size = obs['next_cat_size'][0]
        info = {
            "score": self.last_score,
            "next_cat_size": cat_size,
            "did_click": 1.0 if did_click else 0.0,
            "is_game_over": False
        }

        max_val_r, _ = self._find_template(full_img, self.template_restart)
        if max_val_r > 0.8:
            display = os.environ.get('DISPLAY', ':0')
            print(f"[{display}] Game Over detected.")
            if self.last_score > self.session_high_score:
                print(f"[{display}] *** NEW HIGH SCORE: {self.last_score} ***")
                self.session_high_score = self.last_score
                cv2.imwrite(f"highscore_{self.last_score}.png", full_img)
            
            self.last_score = 0
            self.low_score_counter = 0
            reward = -1000.0 if did_click else 0.0
            info["is_game_over"] = True
            return obs, reward, True, False, info

        # --- Robust Score Logic ---
        current_score = self._read_score(full_img)
        reward = 0.0
        
        if current_score != -1:
            if current_score > self.last_score:
                self.low_score_counter = 0
                if current_score - self.last_score < 500:
                    reward = float(current_score - self.last_score)
                    self.last_score = current_score
            elif current_score < self.last_score:
                self.low_score_counter += 1
                if self.low_score_counter >= 5:
                    display = os.environ.get('DISPLAY', ':0')
                    print(f"[{display}] Score correction: {self.last_score} -> {current_score}")
                    self.last_score = current_score
                    self.low_score_counter = 0
            else:
                self.low_score_counter = 0
        else:
            reward = -0.1
        
        if not did_click:
            wait_penalty = -0.01 * (1.1 ** self.consecutive_waits)
            reward += max(wait_penalty, -1.0)
        
        display = os.environ.get('DISPLAY', ':0')
        print(f"[{display}] Step: {self.step_count:<6} | Action: {action_str:<15} | Score: {self.last_score:<6} | Reward: {reward:>8.2f} | Next Cat Size: {cat_size:.2f}")

        return obs, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_score = 0
        self.low_score_counter = 0
        self.step_count = 0
        self.last_click_time = time.time()
        self.consecutive_waits = 0
        display = os.environ.get('DISPLAY', ':0')
        print(f"\n[{display}] --- Resetting Environment: Verifying Game State ---")
        
        while True:
            img = np.array(self.sct.grab(self.game_region))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            if self.template_ng_play is not None:
                max_val_ng, max_loc_ng = self._find_template(img, self.template_ng_play)
                if max_val_ng > 0.8:
                    print(f"[{display}] State: NEWGROUNDS OVERLAY. Clicking Play...")
                    self._click_template(max_loc_ng, self.template_ng_play)
                    time.sleep(5)
                    continue

            max_val_r, max_loc_r = self._find_template(img, self.template_restart)
            if max_val_r > 0.8:
                print(f"[{display}] State: GAME OVER. Clicking Restart...")
                self._click_template(max_loc_r, self.template_restart)
                time.sleep(2)
                continue

            max_val_p, max_loc_p = self._find_template(img, self.template_play)
            if max_val_p > 0.8:
                if self.template_music is not None and not self.music_muted:
                    max_val_m, max_loc_m = self._find_template(img, self.template_music)
                    if max_val_m > 0.8:
                        print(f"[{display}] State: MENU. Muting Music...")
                        self._click_template(max_loc_m, self.template_music)
                        self.music_muted = True
                        time.sleep(0.5)
                        img = np.array(self.sct.grab(self.game_region))
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                print(f"[{display}] State: MENU. Clicking Play...")
                self._click_template(max_loc_p, self.template_play)
                time.sleep(2)
                continue

            max_val_b, _ = self._find_template(img, self.template_empty_board)
            if max_val_b > 0.8 or (max_val_r < 0.8 and max_val_p < 0.8):
                print(f"[{display}] State: ACTIVE GAME. Starting training.")
                initial_score = self._read_score(img)
                self.last_score = initial_score if initial_score != -1 else 0
                return self._get_observation(), {}

            print(f"[{display}] State: Unknown. Waiting for a known state...")
            time.sleep(1)

    def render(self): pass
    def close(self): pass
