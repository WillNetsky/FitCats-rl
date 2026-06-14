import os
import re
import numpy as np
import cv2

def hex_to_bgr(hex_str):
    # Hex is usually RRGGBB in filenames? Or BGR?
    # debug_cat_count.py uses: f"{dominant_color[2]:02x}{dominant_color[1]:02x}{dominant_color[0]:02x}"
    # dominant_color is BGR (opencv default).
    # So index 2 is Red, 1 is Green, 0 is Blue.
    # The string is RRGGBB.
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)

def analyze():
    template_dir = "./cat_templates"
    stats = {} # {id: {'radii': [], 'colors': []}}

    print(f"Analyzing templates in {template_dir}...")

    for item in os.listdir(template_dir):
        item_path = os.path.join(template_dir, item)
        if os.path.isdir(item_path) and item.startswith("cat_"):
            try:
                cat_id = int(item.split("_")[1])
                if cat_id not in stats:
                    stats[cat_id] = {'radii': [], 'colors': []}
                
                for filename in os.listdir(item_path):
                    if filename.endswith(".png"):
                        # Parse filename: cat_{id}_r{radius}_{hex}.png
                        # Note: sometimes id in filename might differ if manually moved, we trust folder structure for ID?
                        # Or trust filename? Let's trust filename for radius/color.
                        
                        radius_match = re.search(r'_r(\d+)_', filename)
                        # Hex is usually at the end before .png
                        # Filename format: cat_0_r34_cce4b2.png or new_cat_r34_cce4b2_uuid.png
                        # Let's look for the hex pattern [0-9a-f]{6}
                        hex_match = re.search(r'_([0-9a-f]{6})', filename)
                        
                        if radius_match and hex_match:
                            r = int(radius_match.group(1))
                            hex_str = hex_match.group(1)
                            color = hex_to_bgr(hex_str)
                            
                            stats[cat_id]['radii'].append(r)
                            stats[cat_id]['colors'].append(color)
            except ValueError:
                pass

    print("\n--- Analysis Results ---")
    max_color_dist_overall = 0
    
    sorted_ids = sorted(stats.keys())
    for cat_id in sorted_ids:
        data = stats[cat_id]
        radii = data['radii']
        colors = data['colors']
        
        if not radii: continue
        
        min_r, max_r = min(radii), max(radii)
        
        # Calculate max distance between any two colors in this group
        max_dist = 0
        if len(colors) > 1:
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    dist = np.linalg.norm(colors[i] - colors[j])
                    if dist > max_dist:
                        max_dist = dist
        
        # Also calculate max distance from the "mean" color of the group
        mean_color = np.mean(colors, axis=0)
        max_dist_from_mean = 0
        for c in colors:
            d = np.linalg.norm(c - mean_color)
            if d > max_dist_from_mean:
                max_dist_from_mean = d

        print(f"Cat {cat_id}:")
        print(f"  Radius Range: {min_r} - {max_r} (Span: {max_r - min_r})")
        print(f"  Templates: {len(colors)}")
        print(f"  Max Color Spread (Pairwise): {max_dist:.2f}")
        print(f"  Max Dist from Mean: {max_dist_from_mean:.2f}")
        
        if max_dist_from_mean > max_color_dist_overall:
            max_color_dist_overall = max_dist_from_mean

    print("\n--- Recommendation ---")
    print(f"Current Threshold: 40")
    print(f"Suggested Threshold (Max Dist from Mean + Buffer): {max_color_dist_overall * 1.2:.2f}")

if __name__ == "__main__":
    analyze()
