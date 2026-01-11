import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import mss
import time
import os
import pytesseract

# --- Relative Calibration Data ---
CLICK_X_MIN_REL = 385
CLICK_X_MAX_REL = 894
SCORE_ROI = {'x': 49, 'y': 58, 'w': 189, 'h': 58}
AGENT_VIEW_ROI = {'x': 357, 'y': 3, 'w': 564, 'h': 711}
NEXT_CAT_ROI = {'x': 1101, 'y': 56, 'w': 118, 'h': 101}
DROP_Y_REL_PERCENT = 0.15

class FitCatsEnv(gym.Env):
    def __init__(self):
        super(FitCatsEnv, self).__init__()
        pyautogui.FAILSAFE = True

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "next_cat_color": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
        })

        self.game_title_template = cv2.imread("game_title.png", cv2.IMREAD_COLOR)
        self.template_play = cv2.imread("template_play.png", cv2.IMREAD_COLOR)
        self.template_restart = cv2.imread("template_restart.png", cv2.IMREAD_COLOR)
        if any(t is None for t in [self.game_title_template, self.template_play, self.template_restart]):
            raise FileNotFoundError("Required template images not found!")

        self.sct = mss.mss()
        self.game_region = self._locate_game_window()
        
        self.last_score = 0
        self.step_count = 0

    def _locate_game_window(self):
        print("Searching for game window on all screens...")
        # Get all monitors, skipping the first one which is the 'all-in-one' virtual screen
        monitors = self.sct.monitors[1:]
        
        for i, monitor in enumerate(monitors):
            print(f"Searching on monitor {i+1}...")
            try:
                # Define the region for pyautogui to search
                search_region = (monitor['left'], monitor['top'], monitor['width'], monitor['height'])
                region = pyautogui.locateOnScreen("game_title.png", confidence=0.8, region=search_region)
                
                if region:
                    print(f"Game window found on monitor {i+1} at: {region}")
                    return {"top": region.top, "left": region.left, "width": region.width, "height": region.height}
            except pyautogui.PyAutoGUIException:
                # This can happen if the image is not found on the monitor
                print(f"Not found on monitor {i+1}.")
                continue
                
        raise Exception("Could not find the game window on ANY screen. Make sure it's visible and not obstructed.")

    def _find_template_and_click(self, img, template):
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            cx = self.game_region['left'] + max_loc[0] + template.shape[1] // 2
            cy = self.game_region['top'] + max_loc[1] + template.shape[0] // 2
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
        img = np.array(self.sct.grab(self.game_region))
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
        click_x_rel = int(norm_action * (CLICK_X_MAX_REL - CLICK_X_MIN_REL) + CLICK_X_MIN_REL)
        click_x_abs = self.game_region['left'] + click_x_rel
        drop_y_abs = self.game_region['top'] + int(self.game_region['height'] * DROP_Y_REL_PERCENT)
        
        pyautogui.click(click_x_abs, drop_y_abs)
        time.sleep(0.5)

        full_img = np.array(self.sct.grab(self.game_region))
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
        
        cat_color = obs['next_cat_color']
        print(f"Step: {self.step_count} | Action: {action[0]:.2f} | Score: {self.last_score} | Reward: {reward:.2f} | Next Cat: {cat_color}")

        return obs, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_score = 0
        self.step_count = 0
        print("Resetting environment...")
        
        while True:
            img = np.array(self.sct.grab(self.game_region))
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
