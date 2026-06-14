import cv2
import numpy as np
import mss
import os
import json
import time
import argparse
import re
import uuid
from utils import start_game
# Import everything from fit_cats_env to ensure consistency
from fit_cats_env import find_candidate_circles, filter_nested_circles, get_dominant_color, load_templates, MAX_CAT_TYPES

def get_dir_state(path):
    """Returns a dictionary of filenames and their modification times, recursive."""
    state = {}
    if not os.path.exists(path): return state
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith('.png'):
                filepath = os.path.join(root, name)
                try:
                    state[filepath] = os.path.getmtime(filepath)
                except OSError: pass
    return state

# --- Helper to process a region ---
def process_region(img, mask, known_colors, not_cat_colors, radius_ranges, template_dir, color_dist_threshold, is_next_cat=False):
    """
    Detects, classifies, and visualizes cats in a given image region.
    Returns: (census_dict, unclassified_count, discarded_count, debug_img, candidate_count, last_patch)
    """
    debug_img = img.copy()
    
    # 1. Find ALL candidates (no filtering yet)
    raw_circles = find_candidate_circles(img, mask)
    
    census = {i: 0 for i in known_colors.keys()}
    unclassified_count = 0
    discarded_count = 0
    last_patch = None
    
    # 2. Filter "Not-a-Cats" FIRST
    valid_cat_circles = []
    
    for (cx, cy, r) in raw_circles:
        cx, cy, r = int(cx), int(cy), int(r)
        
        if r < 20:
            patch_radius = int(r * 0.90)
        else:
            patch_radius = int(r * 0.60)
            
        x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
        x2, y2 = min(img.shape[1], cx + patch_radius), min(img.shape[0], cy + patch_radius)
        patch = img[y1:y2, x1:x2]
        
        dominant_color = get_dominant_color(patch)
        
        is_not_a_cat = False
        if dominant_color is not None:
            for not_cat_color in not_cat_colors:
                if np.linalg.norm(dominant_color.astype(np.float32) - not_cat_color.astype(np.float32)) < color_dist_threshold:
                    is_not_a_cat = True
                    break
        
        if is_not_a_cat:
            discarded_count += 1
            cv2.circle(debug_img, (cx, cy), r, (100, 100, 100), 3)
            cv2.putText(debug_img, "X", (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        else:
            valid_cat_circles.append((cx, cy, r, dominant_color, patch))

    # 3. Filter Nested Circles on the REMAINING candidates
    circles_to_filter = np.array([(c[0], c[1], c[2]) for c in valid_cat_circles], dtype=np.uint16) if valid_cat_circles else []
    
    if len(circles_to_filter) > 0:
        final_circles = filter_nested_circles(circles_to_filter)
    else:
        final_circles = []

    # 4. Classify the Survivors
    for (cx, cy, r) in final_circles:
        match = None
        for vc in valid_cat_circles:
            if vc[0] == cx and vc[1] == cy and vc[2] == r:
                match = vc
                break
        
        if not match: continue
        
        _, _, _, dominant_color, patch = match
        last_patch = patch

        if dominant_color is not None:
            best_match_id = -1
            min_dist = color_dist_threshold
            
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
            
            if best_match_id != -1:
                census[best_match_id] = census.get(best_match_id, 0) + 1
                viz_color = tuple(map(int, known_colors[best_match_id][0]['color']))
                cv2.circle(debug_img, (cx, cy), r, viz_color, 3)
                cv2.putText(debug_img, str(best_match_id), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
                # --- KEY FIX: Save variant if not a perfect match ---
                PERFECT_MATCH_DIST = 20
                if min_dist > PERFECT_MATCH_DIST:
                    hex_color = f"{dominant_color[2]:02x}{dominant_color[1]:02x}{dominant_color[0]:02x}"
                    if patch.shape[0] > 0 and patch.shape[1] > 0:
                        color_block = np.full(patch.shape, tuple(map(int, dominant_color)), np.uint8)
                        composite_template = cv2.hconcat([patch, color_block])
                        
                        unique_id = str(uuid.uuid4())[:8]
                        # Save directly to the correct cat folder!
                        save_filename = f"cat_{best_match_id}_r{r}_{hex_color}_{unique_id}.png"
                        save_dir = os.path.join(template_dir, f"cat_{best_match_id}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, save_filename)
                        
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, composite_template)
                            print(f"\nSaved variant for Cat {best_match_id}! Dist: {min_dist:.2f}, Path: {save_path}")

            else:
                # Unclassified (New Cat)
                unclassified_count += 1
                hex_color = f"{dominant_color[2]:02x}{dominant_color[1]:02x}{dominant_color[0]:02x}"
                
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    color_block = np.full(patch.shape, tuple(map(int, dominant_color)), np.uint8)
                    composite_template = cv2.hconcat([patch, color_block])
                    
                    unique_id = str(uuid.uuid4())[:8]
                    save_filename = f"new_cat_r{r}_{hex_color}_{unique_id}.png"
                    save_path = os.path.join(template_dir, save_filename)
                    
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path, composite_template)
                        print(f"\nDiscovered new cat! Radius: {r}, Saved to: {save_path}")
                
                cv2.circle(debug_img, (cx, cy), r, (0, 0, 0), 2)
        else:
            unclassified_count += 1
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                unique_id = str(uuid.uuid4())[:8]
                save_filename = f"failed_color_cat_r{r}_{unique_id}.png"
                save_path = os.path.join(template_dir, save_filename)
                cv2.imwrite(save_path, patch)
                # print(f"\nSaved failed color cat! Radius: {r}, Saved to: {save_path}")

            cv2.circle(debug_img, (cx, cy), r, (0, 0, 0), 2)

    return census, unclassified_count, discarded_count, debug_img, len(raw_circles), last_patch

def check_color_conflicts(known_colors, not_cat_colors, threshold):
    """Checks for dangerous proximity between valid cats and not-a-cats."""
    print("\n--- Checking Color Conflicts ---")
    min_dist = float('inf')
    conflict_pair = None

    for t_id, templates in known_colors.items():
        for template in templates:
            cat_color = template['color'].astype(np.float32)
            for not_cat_color in not_cat_colors:
                dist = np.linalg.norm(cat_color - not_cat_color.astype(np.float32))
                if dist < min_dist:
                    min_dist = dist
                    # Note: not_cat_colors is just a list of arrays now, so we don't have filenames
                    conflict_pair = (t_id, template.get('filename', 'unknown'), "unknown_not_cat")

    if conflict_pair:
        t_id, cat_file, nc_file = conflict_pair
        status = "CRITICAL" if min_dist < threshold else "WARNING" if min_dist < threshold * 1.5 else "SAFE"
        print(f"[{status}] Closest Match: Cat ID {t_id} ({cat_file}) vs Not-a-Cat. Dist: {min_dist:.2f} (Threshold: {threshold})")
    else:
        print("No conflicts found (or no data).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=str, default=":99", help="The sandbox display ID")
    parser.add_argument("--record", type=str, help="Record actions to the specified file.")
    args = parser.parse_args()
    
    print("=== Cat Color Classification and Discovery Tool ===")
    
    main_display = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = args.sandbox
    import pyautogui
    if main_display:
        os.environ["DISPLAY"] = main_display
    else:
        if "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

    if not os.path.exists("calibration_data.json"):
        print("Error: calibration_data.json not found.")
        return
    with open("calibration_data.json", "r") as f:
        calib = json.load(f)

    TEMPLATE_DIR = "./cat_templates"
    # --- KEY FIX: Unpack radius_ranges ---
    known_colors, not_cat_colors, radius_ranges = load_templates(TEMPLATE_DIR)
    color_dist_threshold = 100 # Updated threshold

    cv2.namedWindow("Cat Color Census")
    cv2.waitKey(1)

    # --- Recording Logic ---
    recorded_actions = []
    recording_start_time = None
    if args.record:
        print(f"--- RECORDING MODE: Actions will be saved to {args.record} ---")
        recording_start_time = time.time()

    def record_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and args.record:
            abs_x = game_region['left'] + agent_roi['x'] + x
            abs_y = game_region['top'] + agent_roi['y'] + y
            timestamp = time.time() - recording_start_time
            recorded_actions.append({'x': abs_x, 'y': abs_y, 't': timestamp})
            print(f"Recorded click at ({abs_x}, {abs_y}) at time {timestamp:.2f}")
            pyautogui.click(abs_x, abs_y)

    cv2.setMouseCallback("Cat Color Census", record_click)

    with mss.mss(display=args.sandbox) as sct:
        game_region = start_game(sct, calib, pyautogui)
        print("\nDetecting cats and classifying by color. New colors will be saved automatically. Press Ctrl+C to quit.")
        
        agent_roi = calib["agent_view_roi"]
        mask = np.zeros((agent_roi['h'], agent_roi['w']), dtype=np.uint8)
        if "playable_polygon" in calib:
            poly_points = calib["playable_polygon"]
            rel_poly_points = [(px - agent_roi['x'], py - agent_roi['y']) for px, py in poly_points]
            pts = np.array(rel_poly_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

        last_reload_time = time.time()
        last_conflict_check_time = time.time()
        dir_state = get_dir_state(TEMPLATE_DIR)

        while True:
            try:
                if time.time() - last_reload_time > 5.0:
                    new_dir_state = get_dir_state(TEMPLATE_DIR)
                    if new_dir_state != dir_state:
                        print("\nTemplate directory changed, reloading...")
                        known_colors, not_cat_colors, radius_ranges = load_templates(TEMPLATE_DIR)
                        dir_state = new_dir_state
                    last_reload_time = time.time()
                
                if time.time() - last_conflict_check_time > 10.0:
                    check_color_conflicts(known_colors, not_cat_colors, color_dist_threshold)
                    last_conflict_check_time = time.time()

                game_img = np.array(sct.grab(game_region))
                game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
                
                play_area_img = game_img_bgr[agent_roi['y']:agent_roi['y']+agent_roi['h'], agent_roi['x']:agent_roi['x']+agent_roi['w']]
                census, unclass, discarded, debug_board, candidates, _ = process_region(play_area_img, mask, known_colors, not_cat_colors, radius_ranges, TEMPLATE_DIR, color_dist_threshold)
                
                next_cat_roi = calib["next_cat_roi"]
                next_cat_img = game_img_bgr[next_cat_roi['y']:next_cat_roi['y']+next_cat_roi['h'], next_cat_roi['x']:next_cat_roi['x']+next_cat_roi['w']]
                census_next, unclass_next, discarded_next, debug_next, candidates_next, next_patch = process_region(next_cat_img, None, known_colors, not_cat_colors, radius_ranges, TEMPLATE_DIR, color_dist_threshold, is_next_cat=True)
                
                for k, v in census_next.items():
                    census[k] = census.get(k, 0) + v
                
                total_unclass = unclass + unclass_next
                total_discarded = discarded + discarded_next
                total_candidates = candidates + candidates_next

                census_str = " | ".join([f"ID {k}: {v}" for k, v in sorted(census.items()) if v > 0])
                
                next_cat_id = "None"
                for k, v in census_next.items():
                    if v > 0:
                        next_cat_id = str(k)
                        break
                
                print(f"\rBoard: [ {census_str} ] | Next: {next_cat_id} | Unclass: {total_unclass}   ", end="")
                
                h_next, w_next = debug_next.shape[:2]
                debug_board[0:h_next, 0:w_next] = debug_next
                cv2.rectangle(debug_board, (0,0), (w_next, h_next), (0, 255, 255), 2)
                
                if next_patch is not None and next_patch.shape[0] > 0 and next_patch.shape[1] > 0:
                    h_patch, w_patch = next_patch.shape[:2]
                    debug_board[h_next:h_next+h_patch, 0:w_patch] = next_patch
                    cv2.rectangle(debug_board, (0, h_next), (w_patch, h_next+h_patch), (255, 0, 255), 1)
                
                cv2.imshow("Cat Color Census", debug_board)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)

            except KeyboardInterrupt:
                print("\nExiting Cat Count Debugger.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    if args.record and recorded_actions:
        with open(args.record, 'w') as f:
            json.dump(recorded_actions, f, indent=4)
        print(f"\nSaved {len(recorded_actions)} actions to {args.record}")

if __name__ == "__main__":
    main()
