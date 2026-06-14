import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
import os
import json
import re
from collections import deque
import datetime
from utils import start_game

# --- Constants ---
MAX_CAT_TYPES = 15
COLOR_DIST_THRESHOLD = 100
IMG_SIZE = 160 
CLICK_COOLDOWN_SECONDS = 0.6

# --- Custom Template-Based OCR Functions ---
def get_digit_templates(template_dir="digit_templates"):
    digit_templates = {}
    if not os.path.exists(template_dir): return digit_templates
    for digit_folder in os.listdir(template_dir):
        if not digit_folder.isdigit(): continue
        digit_path = os.path.join(template_dir, digit_folder)
        if os.path.isdir(digit_path):
            templates = [cv2.imread(os.path.join(digit_path, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(digit_path)]
            digit_templates[digit_folder] = [t for t in templates if t is not None]
    return digit_templates

def find_contours(score_img):
    gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [], None
    min_w, min_h = 5, 10
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > min_w and cv2.boundingRect(c)[3] > min_h]
    return sorted(bounding_boxes, key=lambda b: b[0]), thresh

def recognize_score_with_templates(score_img, digit_templates):
    sorted_boxes, thresh = find_contours(score_img)
    if not sorted_boxes: return ""
    recognized_score = ""
    for (x, y, w, h) in sorted_boxes:
        padding = 5
        digit_roi = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
        best_overall_score, best_overall_digit = 0.5, ""
        for digit_str, templates in digit_templates.items():
            best_score_for_this_digit = 0.0
            for template_img in templates:
                if digit_roi.shape[0] < template_img.shape[0] or digit_roi.shape[1] < template_img.shape[1]: continue
                res = cv2.matchTemplate(digit_roi, template_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_score_for_this_digit: best_score_for_this_digit = max_val
            if best_score_for_this_digit > best_overall_score:
                best_overall_score, best_overall_digit = best_score_for_this_digit, digit_str
        if best_overall_digit: recognized_score += best_overall_digit
    return recognized_score

# --- Cat Detection & Classification Functions (Ported from debug_cat_count.py) ---

def get_dominant_color(img, k=3):
    if img is None or img.shape[0] < 5 or img.shape[1] < 5:
        return None
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    sorted_centers = centers[np.argsort(-counts)]
    return np.uint8(sorted_centers[0])

def circle_intersection_area(c1, c2):
    x1, y1, r1 = float(c1[0]), float(c1[1]), float(c1[2])
    x2, y2, r2 = float(c2[0]), float(c2[1]), float(c2[2])
    d = np.linalg.norm([x1 - x2, y1 - y2])
    if d >= r1 + r2: return 0.0
    if d <= abs(r1 - r2): return np.pi * min(r1, r2)**2
    r1_sq, r2_sq = r1**2, r2**2
    alpha = np.arccos((r1_sq + d**2 - r2_sq) / (2 * r1 * d))
    beta = np.arccos((r2_sq + d**2 - r1_sq) / (2 * r2 * d))
    return r1_sq * alpha + r2_sq * beta - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

def find_candidate_circles(img, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        masked_gray = cv2.bitwise_and(gray, mask)
        blurred = cv2.GaussianBlur(masked_gray, (5, 5), 1.5)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Pass 1: Standard (15-50)
    circles_mid = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                               param1=50, param2=40, minRadius=15, maxRadius=50)
    
    # Pass 2: Small (8-20)
    circles_small = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=30, minRadius=15, maxRadius=20)

    # Pass 3: Large (50-140)
    circles_large = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=45, minRadius=50, maxRadius=140)
                               
    # Pass 4: Gigantic (140-180)
    circles_giant = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=140, maxRadius=180)
    
    all_circles = []
    if circles_mid is not None: all_circles.extend(circles_mid[0, :])
    if circles_small is not None: all_circles.extend(circles_small[0, :])
    if circles_large is not None: all_circles.extend(circles_large[0, :])
    if circles_giant is not None: all_circles.extend(circles_giant[0, :])

    if not all_circles: return []
    circles = np.uint16(np.around(all_circles))
    return circles

def filter_nested_circles(circles):
    discarded_indices = set()
    for i, c1 in enumerate(circles):
        if i in discarded_indices: continue
        for j, c2 in enumerate(circles):
            if i == j or j in discarded_indices: continue
            smaller, larger = (c1, c2) if c1[2] < c2[2] else (c2, c1)
            smaller_idx, larger_idx = (i, j) if c1[2] < c2[2] else (j, i)
            intersection = circle_intersection_area(smaller, larger)
            area_smaller = np.pi * float(smaller[2])**2
            if area_smaller == 0: continue
            if intersection / area_smaller > 0.8:
                discarded_indices.add(smaller_idx)
    return [c for i, c in enumerate(circles) if i not in discarded_indices]

def load_templates(template_dir):
    known_colors = {} 
    not_cat_colors = []
    radius_ranges = {} 
    
    if not os.path.exists(template_dir): return known_colors, not_cat_colors, radius_ranges
    not_a_cat_dir = os.path.join(template_dir, "not_a_cat")

    # 1. Load Truth Folders
    for item in os.listdir(template_dir):
        item_path = os.path.join(template_dir, item)
        if os.path.isdir(item_path) and item.startswith("cat_"):
            try:
                cat_id = int(item.split("_")[1])
                if cat_id not in known_colors: known_colors[cat_id] = []
                
                radii = []
                for filename in os.listdir(item_path):
                    if filename.endswith(".png"):
                        img = cv2.imread(os.path.join(item_path, filename))
                        if img is None: continue
                        color = get_dominant_color(img)
                        radius_match = re.search(r'_r(\d+)_', filename)
                        radius = int(radius_match.group(1)) if radius_match else 30
                        if color is not None:
                            known_colors[cat_id].append({'color': color, 'radius': radius})
                            radii.append(radius)
                
                if radii:
                    min_r = min(radii) - 3
                    max_r = max(radii) + 3
                    radius_ranges[cat_id] = (min_r, max_r)
                    
            except ValueError: pass 

    # 2. Load Not-a-Cat
    if os.path.exists(not_a_cat_dir):
        for filename in os.listdir(not_a_cat_dir):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(not_a_cat_dir, filename))
                if img is None: continue
                color = get_dominant_color(img)
                if color is not None:
                    not_cat_colors.append(color)
    return known_colors, not_cat_colors, radius_ranges

def get_cat_census_and_next(board_img, next_cat_img, mask, known_colors, not_cat_colors, radius_ranges):
    """
    Analyzes the board and next cat image to return the census, next cat type, and pile height.
    """
    # --- Process Board ---
    raw_circles = find_candidate_circles(board_img, mask)
    valid_cat_circles = []
    
    for (cx, cy, r) in raw_circles:
        cx, cy, r = int(cx), int(cy), int(r)
        if r < 15: continue 

        if r < 20: patch_radius = int(r * 0.90)
        else: patch_radius = int(r * 0.60)
        
        x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
        x2, y2 = min(board_img.shape[1], cx + patch_radius), min(board_img.shape[0], cy + patch_radius)
        patch = board_img[y1:y2, x1:x2]
        
        dominant_color = get_dominant_color(patch)
        is_not_a_cat = False
        if dominant_color is not None:
            for not_cat_color in not_cat_colors:
                if np.linalg.norm(dominant_color.astype(np.float32) - not_cat_color.astype(np.float32)) < COLOR_DIST_THRESHOLD:
                    is_not_a_cat = True
                    break
        
        if not is_not_a_cat:
            valid_cat_circles.append((cx, cy, r, dominant_color))

    circles_to_filter = np.array([(c[0], c[1], c[2]) for c in valid_cat_circles], dtype=np.uint16) if valid_cat_circles else []
    final_circles = filter_nested_circles(circles_to_filter) if len(circles_to_filter) > 0 else []

    census = np.zeros(MAX_CAT_TYPES, dtype=np.float32)
    min_y = board_img.shape[0] # Default to bottom of screen (empty)
    
    claw_zone_y = int(board_img.shape[0] * 0.15)

    for (cx, cy, r) in final_circles:
        cat_top = int(cy) - int(r)
        
        if cy > claw_zone_y:
            if cat_top < min_y:
                min_y = cat_top

        match = None
        for vc in valid_cat_circles:
            if vc[0] == cx and vc[1] == cy and vc[2] == r:
                match = vc
                break
        if not match: continue
        _, _, _, dominant_color = match

        if dominant_color is not None:
            best_match_id = -1
            min_dist = COLOR_DIST_THRESHOLD
            for t_id, templates in known_colors.items():
                if t_id in radius_ranges:
                    min_r, max_r = radius_ranges[t_id]
                    if not (min_r <= r <= max_r):
                        continue 

                for template in templates:
                    dist = np.linalg.norm(dominant_color.astype(np.float32) - template['color'].astype(np.float32))
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = t_id
            
            if best_match_id != -1 and best_match_id < MAX_CAT_TYPES:
                census[best_match_id] += 1
    
    pile_height = 1.0 - (min_y / board_img.shape[0])
    pile_height = max(0.0, min(1.0, pile_height))

    # --- Process Next Cat ---
    next_cat_type = -1 
    
    raw_circles_next = find_candidate_circles(next_cat_img)
    valid_next_circles = []
    for (cx, cy, r) in raw_circles_next:
        cx, cy, r = int(cx), int(cy), int(r)
        if r < 15: continue

        if r < 20: patch_radius = int(r * 0.90)
        else: patch_radius = int(r * 0.60)
        x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
        x2, y2 = min(next_cat_img.shape[1], cx + patch_radius), min(next_cat_img.shape[0], cy + patch_radius)
        patch = next_cat_img[y1:y2, x1:x2]
        dominant_color = get_dominant_color(patch)
        
        is_not_a_cat = False
        if dominant_color is not None:
            for not_cat_color in not_cat_colors:
                if np.linalg.norm(dominant_color.astype(np.float32) - not_cat_color.astype(np.float32)) < COLOR_DIST_THRESHOLD:
                    is_not_a_cat = True
                    break
        if not is_not_a_cat:
             valid_next_circles.append((cx, cy, r, dominant_color))

    if valid_next_circles:
        best_next_cat = max(valid_next_circles, key=lambda c: c[2])
        _, _, _, dominant_color = best_next_cat
        
        if dominant_color is not None:
            best_match_id = -1
            min_dist = COLOR_DIST_THRESHOLD
            for t_id, templates in known_colors.items():
                if t_id in radius_ranges:
                    min_r, max_r = radius_ranges[t_id]
                    if not (min_r <= best_next_cat[2] <= max_r):
                        continue

                for template in templates:
                    dist = np.linalg.norm(dominant_color.astype(np.float32) - template['color'].astype(np.float32))
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = t_id
            
            if best_match_id != -1 and best_match_id < MAX_CAT_TYPES:
                next_cat_type = best_match_id

    return census, next_cat_type, pile_height

# --- Main Environment Class ---
class FitCatsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super(FitCatsEnv, self).__init__()
        
        import pyautogui
        self.pyautogui = pyautogui
        self.pyautogui.FAILSAFE = True
        
        self.render_mode = "rgb_array"
        self.model_name = os.environ.get("FITCATS_MODEL_NAME", "default")
        
        if not os.path.exists("calibration_data.json"): raise FileNotFoundError("calibration_data.json not found! Run setup_agent.py first.")
        with open("calibration_data.json", "r") as f: self.calib = json.load(f)
        
        # Load Templates
        self.digit_templates = get_digit_templates()
        self.known_colors, self.not_cat_colors, self.radius_ranges = load_templates("./cat_templates")
        
        if "playable_polygon" not in self.calib or "agent_view_roi" not in self.calib:
            raise ValueError("Calibration data missing 'playable_polygon' or 'agent_view_roi'. Run setup_agent.py.")

        agent_view_roi = self.calib["agent_view_roi"]
        poly_points = self.calib["playable_polygon"]
        self.cat_mask = np.zeros((agent_view_roi['h'], agent_view_roi['w']), dtype=np.uint8)
        rel_poly_points = [(px - agent_view_roi['x'], py - agent_view_roi['y']) for px, py in poly_points]
        pts = np.array(rel_poly_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.cat_mask, [pts], 255)

        self.action_space = spaces.MultiDiscrete([IMG_SIZE, 2])
        
        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "next_cat_type": spaces.Box(low=0, high=1, shape=(MAX_CAT_TYPES,), dtype=np.float32), 
            "time_since_click": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "cat_census": spaces.Box(low=0, high=100, shape=(MAX_CAT_TYPES,), dtype=np.float32),
            "pile_height": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.template_restart = cv2.imread("template_restart.png", cv2.IMREAD_COLOR)
        self.template_empty_board = cv2.imread("template_empty_board.png", cv2.IMREAD_COLOR)
        
        if any(t is None for t in [self.template_restart, self.template_empty_board]): raise FileNotFoundError("Required template images not found! Run setup_agent.py.")
        self.sct = mss.mss()
        self.game_region = start_game(self.sct, self.calib, self.pyautogui)
        
        self.last_score, self.session_high_score, self.step_count, self.last_click_time, self.consecutive_waits, self.low_score_counter, self.music_muted = 0, 0, 0, time.time(), 0, 0, False
        self.last_obs_img = None
        self.last_cat_census = np.zeros(MAX_CAT_TYPES, dtype=np.float32)
        
        self.click_cooldown = 0.0 
        # --- KEY FIX: Track episode clicks ---
        self.episode_clicks = 0
        
        os.makedirs("highscores", exist_ok=True)

        # --- Replay Mode ---
        self.replay_actions = None
        self.replay_start_time = None
        self.replay_idx = 0
        replay_file = os.environ.get("FITCATS_REPLAY_FILE")
        if replay_file:
            if not os.path.exists(replay_file):
                raise FileNotFoundError(f"Replay file not found: {replay_file}")
            with open(replay_file, 'r') as f:
                self.replay_actions = json.load(f)
            print(f"Loaded {len(self.replay_actions)} actions from replay file: {replay_file}")


    def _find_template(self, img, template):
        if template is None or img.shape[0] < template.shape[0] or img.shape[1] < template.shape[1]: return 0, (0,0)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    def _click_template(self, max_loc, template):
        cx, cy = self.game_region['left'] + max_loc[0] + template.shape[1] // 2, self.game_region['top'] + max_loc[1] + template.shape[0] // 2
        self.pyautogui.click(cx, cy)

    def _read_score(self, img):
        try:
            roi = self.calib["score_roi"]
            score_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
            score_text = recognize_score_with_templates(score_img, self.digit_templates)
            return int(score_text) if score_text else -1
        except (ValueError, TypeError): return -1

    def _update_highscores(self, score, cats, img):
        display = os.environ.get('DISPLAY', ':0')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        files_to_update = ["highscores.json"]
        if self.model_name != "default":
            files_to_update.append(f"highscores_{self.model_name}.json")
            
        for filename in files_to_update:
            lock_file = filename + ".lock"
            retries = 10
            while os.path.exists(lock_file) and retries > 0:
                time.sleep(0.1)
                retries -= 1
                
            try:
                with open(lock_file, 'w') as f: f.write("locked")
                
                data = {"high_score": 0, "max_cats": 0, "history": []}
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r') as f:
                            loaded = json.load(f)
                            data["high_score"] = loaded.get("global_high_score", loaded.get("high_score", 0))
                            data["max_cats"] = loaded.get("global_max_cats", loaded.get("max_cats", 0))
                            data["history"] = loaded.get("history", [])
                    except json.JSONDecodeError: pass

                updated = False
                
                folder_name = "global" if filename == "highscores.json" else self.model_name
                save_dir = os.path.join("highscores", folder_name)
                os.makedirs(save_dir, exist_ok=True)

                if score > data["high_score"]:
                    data["high_score"] = score
                    img_name = os.path.join(save_dir, f"score_{score}_{timestamp}.png")
                    cv2.imwrite(img_name, img)
                    data["history"].append({"type": "score", "value": score, "agent": display, "time": timestamp, "file": img_name})
                    updated = True
                    
                if cats > data["max_cats"]:
                    data["max_cats"] = int(cats)
                    img_name = os.path.join(save_dir, f"cats_{int(cats)}_{timestamp}.png")
                    cv2.imwrite(img_name, img)
                    data["history"].append({"type": "cats", "value": int(cats), "agent": display, "time": timestamp, "file": img_name})
                    updated = True
                
                if updated:
                    save_data = {
                        "global_high_score" if filename == "highscores.json" else "high_score": data["high_score"],
                        "global_max_cats" if filename == "highscores.json" else "max_cats": data["max_cats"],
                        "history": data["history"]
                    }
                    with open(filename, 'w') as f:
                        json.dump(save_data, f, indent=4)
                        
            except Exception as e:
                print(f"Error updating {filename}: {e}")
            finally:
                try:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                except FileNotFoundError:
                    pass 

    def _get_observation(self):
        img = np.array(self.sct.grab(self.game_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.last_obs_img = img
        
        roi = self.calib["agent_view_roi"]
        board_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
        board_obs = cv2.resize(board_img, (IMG_SIZE, IMG_SIZE))
        
        roi_next = self.calib["next_cat_roi"]
        next_cat_img = img[roi_next['y']:roi_next['y']+roi_next['h'], roi_next['x']:roi_next['x']+roi_next['w']]
        
        time_delta = time.time() - self.last_click_time
        
        cat_census, next_cat_type, pile_height = get_cat_census_and_next(board_img, next_cat_img, self.cat_mask, self.known_colors, self.not_cat_colors, self.radius_ranges)
        
        next_cat_one_hot = np.zeros(MAX_CAT_TYPES, dtype=np.float32)
        if next_cat_type != -1 and next_cat_type < MAX_CAT_TYPES:
            next_cat_one_hot[next_cat_type] = 1.0

        return {
            "board": board_obs, 
            "next_cat_type": next_cat_one_hot,
            "time_since_click": np.array([time_delta], dtype=np.float32),
            "cat_census": cat_census,
            "pile_height": np.array([pile_height], dtype=np.float32)
        }

    def step(self, action):
        self.step_count += 1
        x_bin, click_trigger = action # Unpack discrete action
        
        attempted_click = click_trigger > 0
        
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
            did_click = False 
        else:
            did_click = attempted_click
        
        old_cat_census = self.last_cat_census.copy()

        if did_click:
            # Map bin to screen coordinate
            min_x, max_x = self.calib["click_x_min_rel"], self.calib["click_x_max_rel"]
            click_x_rel = int((x_bin / (IMG_SIZE - 1)) * (max_x - min_x) + min_x)
            
            click_x_abs = self.game_region['left'] + click_x_rel
            drop_y_abs = self.game_region['top'] + int(self.game_region['height'] * 0.25)
            
            # Clamp coordinates to safe range
            click_x_abs = max(0, min(3000, click_x_abs))
            drop_y_abs = max(0, min(3000, drop_y_abs))
            
            self.pyautogui.moveTo(click_x_abs, drop_y_abs, duration=0.1)
            self.pyautogui.mouseDown()
            time.sleep(0.1)
            self.pyautogui.mouseUp()
            
            self.last_click_time = time.time()
            self.consecutive_waits = 0
            
            self.click_cooldown = 6 
            # --- KEY FIX: Increment episode clicks ---
            self.episode_clicks += 1
            
            time.sleep(0.1)
        else:
            self.consecutive_waits += 1
            time.sleep(0.1)

        obs = self._get_observation()
        new_cat_census = obs["cat_census"]
        self.last_cat_census = new_cat_census.copy()
        
        max_cat_type = -1
        for i in range(MAX_CAT_TYPES - 1, -1, -1):
            if new_cat_census[i] > 0:
                max_cat_type = i
                break
        
        old_total_cats = np.sum(old_cat_census)
        new_total_cats = np.sum(new_cat_census)
        
        cat_diff = old_total_cats - new_total_cats
        
        if did_click:
            reward = -5.0 # Click Cost
        else:
            reward = 0.0
            
        if cat_diff > 0:
            reward += cat_diff * 50.0 # Merge Reward
        
        if not did_click:
            if self.consecutive_waits > 50:
                reward += -0.05 * (1.1 ** (self.consecutive_waits - 50))
        
        # Check for game over after reward calculation
        max_val_r, _ = self._find_template(self.last_obs_img, self.template_restart)
        if max_val_r > 0.8:
            display = os.environ.get('DISPLAY', ':0')
            if self.last_score > self.session_high_score:
                self.session_high_score = self.last_score
            
            info = {
                "game/score": self.last_score,
                "game/step_count": self.step_count,
                "game/did_click": 1.0 if did_click else 0.0,
                "game/attempted_click": 1.0 if attempted_click else 0.0, 
                "game/successful_click": 0.0, 
                "game/cat_count": np.sum(new_cat_census),
                "game/max_cat_type": max_cat_type,
                "game/action_x": x_bin,
                "game/final_score": self.last_score,
                "game/final_cat_count": np.sum(new_cat_census),
                "game/final_max_cat_type": max_cat_type,
                "game/final_pile_height": obs["pile_height"][0],
                "game/final_clicks": self.episode_clicks, # Added
                "is_game_over": True
            }
            
            reward = -1000.0
            return obs, reward, True, False, info

        current_score = self._read_score(self.last_obs_img)
        
        score_increased = 0.0
        if current_score > self.last_score:
            score_increased = 1.0
            reward += (current_score - self.last_score)
            self.last_score = current_score
            
        info = {
            "game/score": self.last_score,
            "game/step_count": self.step_count,
            "game/did_click": 1.0 if did_click else 0.0,
            "game/attempted_click": 1.0 if attempted_click else 0.0, 
            "game/successful_click": score_increased, 
            "game/cat_count": np.sum(new_cat_census),
            "game/max_cat_type": max_cat_type,
            "game/action_x": x_bin,
            "is_game_over": False
        }
        
        return obs, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- KEY FIX: Print stats on reset ---
        display = os.environ.get('DISPLAY', ':0')
        if self.step_count > 0: # Only print if we actually played
            print(f"[{display}] Game Over | Score: {self.last_score} | Clicks: {self.episode_clicks} | Steps: {self.step_count}")

        self.last_score, self.low_score_counter, self.step_count = 0, 0, 0
        self.last_click_time = time.time()
        self.consecutive_waits = 0
        self.click_cooldown = 0 
        self.episode_clicks = 0 # Reset clicks
        
        print(f"[{display}] --- Resetting Environment ---")
        
        if self.replay_actions:
            print(f"[{display}] --- Executing Replay ---")
            self.replay_start_time = time.time()
            self.replay_idx = 0
            
            while self.replay_idx < len(self.replay_actions):
                action = self.replay_actions[self.replay_idx]
                current_replay_time = time.time() - self.replay_start_time
                
                if current_replay_time >= action['t']:
                    self.pyautogui.click(action['x'], action['y'])
                    self.replay_idx += 1
                    time.sleep(0.1) 
                else:
                    time.sleep(0.05) 
            
            print(f"[{display}] --- Replay Finished ---")
            time.sleep(2) 

        while True:
            img = np.array(self.sct.grab(self.game_region))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            max_val_r, max_loc_r = self._find_template(img, self.template_restart)
            if max_val_r > 0.8:
                self._click_template(max_loc_r, self.template_restart)
                time.sleep(2)
                continue
            
            max_val_b, _ = self._find_template(img, self.template_empty_board)
            if max_val_b > 0.8:
                initial_obs = self._get_observation()
                self.last_score = self._read_score(self.last_obs_img) if self._read_score(self.last_obs_img) != -1 else 0
                self.last_cat_census = initial_obs["cat_census"].copy()
                return initial_obs, {}
            
            print(f"[{display}] Waiting for game to be ready for reset...")
            time.sleep(1)

    def render(self):
        if self.render_mode == "rgb_array" and self.last_obs_img is not None:
            return self.last_obs_img
        return None

    def close(self): pass
