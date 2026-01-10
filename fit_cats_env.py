import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import mss
import time
import os
import pytesseract

# --- Hardcoded Calibration Data ---
GAME_REGION = {'top': 446, 'left': 1035, 'width': 742, 'height': 481}
CLICK_X_MIN = 1255
CLICK_X_MAX = 1552
SCORE_ROI = {'x': 27, 'y': 68, 'w': 109, 'h': 32}
AGENT_VIEW_ROI = {'x': 205, 'y': 32, 'w': 327, 'h': 417}
NEXT_CAT_ROI = {'x': 639, 'y': 67, 'w': 66, 'h': 57}
DROP_Y = GAME_REGION['top'] + int(GAME_REGION['height'] * 0.15)

class FitCatsEnv(gym.Env):
    def __init__(self):
        super(FitCatsEnv, self).__init__()
        pyautogui.FAILSAFE = True

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "next_cat_color": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
        })

        self.template_play = cv2.imread("template_play.png")
        self.template_restart = cv2.imread("template_restart.png")
        if self.template_play is None or self.template_restart is None:
            raise FileNotFoundError("Templates not found! Run analyze_game_ui.py first.")

        self.sct = mss.mss()
        self.last_score = 0
        self.step_count = 0

    def _find_template_and_click(self, img, template):
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            cx = GAME_REGION['left'] + max_loc[0] + template.shape[1] // 2
            cy = GAME_REGION['top'] + max_loc[1] + template.shape[0] // 2
            pyautogui.click(cx, cy)
            return True
        return False

    def _read_score(self, img):
        try:
            sx, sy, sw, sh = SCORE_ROI['x'], SCORE_ROI['y'], SCORE_ROI['w'], SCORE_ROI['h']
            score_img = img[sy:sy+sh, sx:sx+sw]
            
            gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            score_text = pytesseract.image_to_string(thresh, config=custom_config)
            return int(score_text.strip())
        except:
            return -1

    def _get_observation(self):
        img = np.array(self.sct.grab(GAME_REGION))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        ax, ay, aw, ah = AGENT_VIEW_ROI['x'], AGENT_VIEW_ROI['y'], AGENT_VIEW_ROI['w'], AGENT_VIEW_ROI['h']
        board_img = img[ay:ay+ah, ax:ax+aw]
        board_obs = cv2.resize(board_img, (84, 84))

        nx, ny, nw, nh = NEXT_CAT_ROI['x'], NEXT_CAT_ROI['y'], NEXT_CAT_ROI['w'], NEXT_CAT_ROI['h']
        next_cat_img = img[ny:ny+nh, nx:nx+nw]
        avg_color = np.array(cv2.mean(next_cat_img)[:3], dtype=np.uint8)

        return {"board": board_obs, "next_cat_color": avg_color}

    def step(self, action):
        self.step_count += 1
        
        norm_action = (action[0] + 1) / 2.0
        click_x = int(norm_action * (CLICK_X_MAX - CLICK_X_MIN) + CLICK_X_MIN)
        pyautogui.click(click_x, DROP_Y)
        time.sleep(0.5)

        full_img = np.array(self.sct.grab(GAME_REGION))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)
        obs = self._get_observation()

        if self._find_template_and_click(full_img, self.template_restart):
            print("Game Over detected.")
            self.last_score = 0
            return obs, -1000.0, True, False, {}

        current_score = self._read_score(full_img)
        reward = 0.0
        if current_score > self.last_score:
            reward = float(current_score - self.last_score)
            self.last_score = current_score
        elif current_score == -1:
            reward = -0.1
        
        # Updated log message
        cat_color = obs['next_cat_color']
        print(f"Step: {self.step_count} | Action: {action[0]:.2f} | Score: {self.last_score} | Reward: {reward:.2f} | Next Cat: {cat_color}")

        return obs, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_score = 0
        self.step_count = 0
        print("\n--- Resetting Environment ---")
        
        while True:
            img = np.array(self.sct.grab(GAME_REGION))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            if self._find_template_and_click(img, self.template_restart):
                print("Found Restart Button. Clicking...")
                time.sleep(2)
                continue
            
            if self._find_template_and_click(img, self.template_play):
                print("Found Play Button. Clicking...")
                time.sleep(2)
                continue
            
            print("Game is active.")
            return self._get_observation(), {}

    def render(self): pass
    def close(self): pass
