import cv2
import numpy as np
import os
import json
import uuid
# Import the actual detection logic from the environment
from fit_cats_env import find_candidate_circles, filter_nested_circles, get_dominant_color, load_templates

def main():
    img_path = "highscores/Merge_Cats/score_2960_2026-01-18_04-37-48.png"
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    print(f"Testing FULL detection pipeline on: {img_path}")
    img = cv2.imread(img_path)
    
    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
        
    agent_roi = calib["agent_view_roi"]
    
    h, w = img.shape[:2]
    if h == calib["game_height"]:
        board_img = img[agent_roi['y']:agent_roi['y']+agent_roi['h'], agent_roi['x']:agent_roi['x']+agent_roi['w']]
    else:
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

    # 1. Run the multi-pass detection
    print("Running find_candidate_circles (4 passes)...")
    raw_circles = find_candidate_circles(board_img, mask)
    print(f"  -> Found {len(raw_circles)} raw candidates.")

    # 2. Run the nested filter
    print("Running filter_nested_circles...")
    if len(raw_circles) > 0:
        final_circles = filter_nested_circles(raw_circles)
    else:
        final_circles = []
        
    print(f"  -> {len(final_circles)} circles remain after filtering.")

    # 3. Load existing templates to check for duplicates
    TEMPLATE_DIR = "./cat_templates"
    known_colors, not_cat_colors = load_templates(TEMPLATE_DIR)
    color_dist_threshold = 40

    # 4. Process and Save
    debug_img = board_img.copy()
    final_circles.sort(key=lambda x: x[2], reverse=True)
    
    print("\nDetected Cats:")
    for i, c in enumerate(final_circles):
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        print(f"  Cat {i}: Radius {r} at ({cx}, {cy})")
        
        # Extract Patch
        if r < 20: patch_radius = int(r * 0.90)
        else: patch_radius = int(r * 0.60)
        
        x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
        x2, y2 = min(board_img.shape[1], cx + patch_radius), min(board_img.shape[0], cy + patch_radius)
        patch = board_img[y1:y2, x1:x2]
        
        dominant_color = get_dominant_color(patch)
        
        if dominant_color is not None:
            # Check if known
            best_match_id = -1
            min_dist = color_dist_threshold
            
            # Debug: Print closest match
            closest_dist = float('inf')
            closest_id = -1
            
            for t_id, templates in known_colors.items():
                for template in templates:
                    dist = np.linalg.norm(dominant_color.astype(np.float32) - template['color'].astype(np.float32))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_id = t_id
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = t_id
            
            if best_match_id == -1:
                # New Cat! Save it.
                hex_color = f"{dominant_color[2]:02x}{dominant_color[1]:02x}{dominant_color[0]:02x}"
                
                patch_radius_t = int(r * 0.90)
                x1_t, y1_t = max(0, cx - patch_radius_t), max(0, cy - patch_radius_t)
                x2_t, y2_t = min(board_img.shape[1], cx + patch_radius_t), min(board_img.shape[0], cy + patch_radius_t)
                template_patch = board_img[y1_t:y2_t, x1_t:x2_t]

                if template_patch.shape[0] > 0 and template_patch.shape[1] > 0:
                    color_block = np.full(template_patch.shape, tuple(map(int, dominant_color)), np.uint8)
                    composite_template = cv2.hconcat([template_patch, color_block])
                    
                    unique_id = str(uuid.uuid4())[:8]
                    save_filename = f"new_cat_r{r}_{hex_color}_{unique_id}.png"
                    save_path = os.path.join(TEMPLATE_DIR, save_filename)
                    
                    cv2.imwrite(save_path, composite_template)
                    print(f"    -> SAVED NEW TEMPLATE: {save_filename}")
            else:
                print(f"    -> Matches Cat ID {best_match_id} (Dist: {closest_dist:.2f})")
        else:
            print("    -> Color detection failed (None)")

        # Visualization
        if r > 100: color = (0, 0, 255)
        elif r > 50: color = (0, 255, 255)
        else: color = (0, 255, 0)
            
        cv2.circle(debug_img, (cx, cy), r, color, 2)
        cv2.putText(debug_img, str(r), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("debug_detection_result.png", debug_img)
    print("\nSaved visualization to 'debug_detection_result.png'")

if __name__ == "__main__":
    main()
