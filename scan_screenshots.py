import cv2
import numpy as np
import os
import json
import re
import uuid
import argparse

# Import detection logic from the environment
from fit_cats_env import find_candidate_circles, get_dominant_color, filter_nested_circles, load_templates, MAX_CAT_TYPES

def process_static_image(img_path, known_colors, not_cat_colors, template_dir, color_dist_threshold=40):
    print(f"Scanning: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("  Error: Could not read image.")
        return

    if not os.path.exists("calibration_data.json"):
        print("  Error: calibration_data.json not found.")
        return
    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
        
    agent_roi = calib["agent_view_roi"]
    
    h, w = img.shape[:2]
    
    if h == calib["game_height"] and w == calib["game_width"]:
        board_img = img[agent_roi['y']:agent_roi['y']+agent_roi['h'], agent_roi['x']:agent_roi['x']+agent_roi['w']]
    elif h == agent_roi['h'] and w == agent_roi['w']:
        board_img = img
    else:
        print(f"  Warning: Image size {w}x{h} does not match expected game or agent view. Attempting to process full image.")
        board_img = img

    mask = np.zeros((board_img.shape[0], board_img.shape[1]), dtype=np.uint8)
    if "playable_polygon" in calib:
        poly_points = calib["playable_polygon"]
        if h == calib["game_height"]:
             rel_poly_points = [(px - agent_roi['x'], py - agent_roi['y']) for px, py in poly_points]
        else:
             rel_poly_points = [(px - agent_roi['x'], py - agent_roi['y']) for px, py in poly_points]

        pts = np.array(rel_poly_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    else:
        mask[:] = 255 

    # --- Detection Logic ---
    raw_circles = find_candidate_circles(board_img, mask)
    
    valid_cat_circles = []
    for (cx, cy, r) in raw_circles:
        cx, cy, r = int(cx), int(cy), int(r)
        if r < 20: patch_radius = int(r * 0.90)
        else: patch_radius = int(r * 0.60)
        
        x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
        x2, y2 = min(board_img.shape[1], cx + patch_radius), min(board_img.shape[0], cy + patch_radius)
        patch = board_img[y1:y2, x1:x2]
        
        dominant_color = get_dominant_color(patch)
        
        is_not_a_cat = False
        if dominant_color is not None:
            for not_cat_color in not_cat_colors:
                if np.linalg.norm(dominant_color.astype(np.float32) - not_cat_color.astype(np.float32)) < color_dist_threshold:
                    is_not_a_cat = True
                    break
        
        if not is_not_a_cat:
            valid_cat_circles.append((cx, cy, r, dominant_color, patch))

    circles_to_filter = np.array([(c[0], c[1], c[2]) for c in valid_cat_circles], dtype=np.uint16) if valid_cat_circles else []
    final_circles = filter_nested_circles(circles_to_filter) if len(circles_to_filter) > 0 else []

    new_cats_found = 0
    
    for (cx, cy, r) in final_circles:
        match = None
        for vc in valid_cat_circles:
            if vc[0] == cx and vc[1] == cy and vc[2] == r:
                match = vc
                break
        if not match: continue
        _, _, _, dominant_color, patch = match

        if dominant_color is not None:
            best_match_id = -1
            min_dist = color_dist_threshold
            
            for t_id, templates in known_colors.items():
                for template in templates:
                    dist = np.linalg.norm(dominant_color.astype(np.float32) - template['color'].astype(np.float32))
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = t_id
            
            if best_match_id == -1:
                # New Discovery!
                hex_color = f"{dominant_color[2]:02x}{dominant_color[1]:02x}{dominant_color[0]:02x}"
                
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    color_block = np.full(patch.shape, tuple(map(int, dominant_color)), np.uint8)
                    composite_template = cv2.hconcat([patch, color_block])
                    
                    unique_id = str(uuid.uuid4())[:8]
                    save_filename = f"new_cat_r{r}_{hex_color}_{unique_id}.png"
                    save_path = os.path.join(template_dir, save_filename)
                    
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path, composite_template)
                        print(f"  -> Found NEW cat! Radius: {r}, Color: {hex_color}")
                        new_cats_found += 1

    if new_cats_found == 0:
        print("  No new cats found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="highscores", help="Base directory to scan")
    # --- KEY FIX: Add model-name argument ---
    parser.add_argument("--model-name", type=str, help="Specific model name to scan (e.g., Cat_Census_v2)")
    args = parser.parse_args()
    
    print("=== Highscore Screenshot Scanner ===")
    
    TEMPLATE_DIR = "./cat_templates"
    known_colors, not_cat_colors = load_templates(TEMPLATE_DIR)
    
    scan_path = args.dir
    if args.model_name:
        scan_path = os.path.join(args.dir, args.model_name)
        if not os.path.exists(scan_path):
            print(f"Error: Model directory not found: {scan_path}")
            return
    
    print(f"Scanning directory: {scan_path}")
    
    for root, _, files in os.walk(scan_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                process_static_image(os.path.join(root, file), known_colors, not_cat_colors, TEMPLATE_DIR)

if __name__ == "__main__":
    main()
