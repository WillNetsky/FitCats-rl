import cv2
import numpy as np
import mss
import os
import json
import time
import argparse
import torch
from stable_baselines3 import PPO
from collections import deque
from scipy.stats import norm
from utils import start_game
from fit_cats_env import get_cat_census_and_next, load_templates, MAX_CAT_TYPES

# NOTE: PyAutoGUI is imported inside main() after setting the DISPLAY variable.

def preprocess_observation(img, calib, known_colors, not_cat_colors, cat_mask):
    """Prepares a raw image into the format the agent expects."""
    roi = calib["agent_view_roi"]
    board_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    board_obs = cv2.resize(board_img, (84, 84))
    
    # Get additional features
    roi_next = calib["next_cat_roi"]
    next_cat_img = img[roi_next['y']:roi_next['y']+roi_next['h'], roi_next['x']:roi_next['x']+roi_next['w']]
    
    cat_census, next_cat_type, pile_height = get_cat_census_and_next(board_img, next_cat_img, cat_mask, known_colors, not_cat_colors)
    
    # Convert to tensors
    board_tensor = torch.tensor(board_obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    next_cat_one_hot = torch.zeros(MAX_CAT_TYPES, dtype=torch.float32)
    if next_cat_type != -1 and next_cat_type < MAX_CAT_TYPES:
        next_cat_one_hot[next_cat_type] = 1.0
        
    cat_census_tensor = torch.tensor(cat_census, dtype=torch.float32)
    pile_height_tensor = torch.tensor([pile_height], dtype=torch.float32)
    
    return board_tensor, next_cat_one_hot, cat_census_tensor, pile_height_tensor

def get_policy_distribution(model, obs_dict):
    """Gets the mean and std of the action distribution."""
    # Ensure inputs are on device and have batch dim
    device = model.device
    for k, v in obs_dict.items():
        if v.dim() == 1: # Vector inputs
            obs_dict[k] = v.unsqueeze(0).to(device)
        elif v.dim() == 3: # Image input (C, H, W)
            obs_dict[k] = v.unsqueeze(0).to(device)
        else:
            obs_dict[k] = v.to(device)

    features = model.policy.extract_features(obs_dict)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    mean_actions = model.policy.action_net(latent_pi)
    
    log_std = model.policy.log_std
    
    return mean_actions[0, 0].item(), mean_actions[0, 1].item(), log_std[0].exp().item()

def generate_saliency_map(model, obs_dict):
    """Generates a saliency map for the given model and stacked frames."""
    device = model.device
    # Ensure inputs are on device and have batch dim
    for k, v in obs_dict.items():
        if v.dim() == 1:
            obs_dict[k] = v.unsqueeze(0).to(device)
        elif v.dim() == 3:
            obs_dict[k] = v.unsqueeze(0).to(device)
        else:
            obs_dict[k] = v.to(device)
            
    obs_dict["board"].requires_grad_()
    
    combined_features = model.policy.features_extractor(obs_dict)
    latent_pi = model.policy.mlp_extractor.forward_actor(combined_features)
    mean_actions = model.policy.action_net(latent_pi)
    
    mean_actions[0, 0].backward(retain_graph=True)
    
    saliency, _ = torch.max(obs_dict["board"].grad.data.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()
    
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    saliency_heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    
    return saliency_heatmap

def gaussian_pdf(x, mean, std):
    """Calculates the Gaussian PDF using numpy."""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def draw_policy_heatmap_bar(img, mean, std, calib):
    """Draws the policy distribution as a colored bar at the top of the clickable area."""
    h, w, _ = img.shape # 84x84
    
    orig_view_w = calib["agent_view_roi"]["w"]
    scale_factor = w / orig_view_w
    agent_view_x = calib["agent_view_roi"]["x"]
    
    orig_click_min_x = calib["click_x_min_rel"] - agent_view_x
    orig_click_max_x = calib["click_x_max_rel"] - agent_view_x
    
    click_min_x = int(orig_click_min_x * scale_factor)
    click_max_x = int(orig_click_max_x * scale_factor)
    click_width = click_max_x - click_min_x
    
    if click_width <= 0: return img

    x_axis = np.linspace(-1, 1, click_width)
    pdf = gaussian_pdf(x_axis, mean, std)
    
    if pdf.max() > 0:
        pdf_normalized = pdf / pdf.max()
    else:
        pdf_normalized = np.zeros_like(pdf)
    
    bar_bgr = np.zeros((1, click_width, 3), dtype=np.uint8)
    
    for i in range(click_width):
        val = pdf_normalized[i]
        if val < 0.5:
            g = int(val * 2 * 255)
            bar_bgr[0, i] = (0, g, 255)
        else:
            r = int((1.0 - val) * 2 * 255)
            bar_bgr[0, i] = (0, 255, r)
            
    bar_height = 5
    bar_y_offset = 10
    
    bar_resized = cv2.resize(bar_bgr, (click_width, bar_height), interpolation=cv2.INTER_NEAREST)
    
    paste_x_start = max(0, click_min_x)
    paste_x_end = min(w, click_max_x)
    
    if paste_x_end > paste_x_start:
        bar_crop_start = paste_x_start - click_min_x
        bar_crop_end = bar_crop_start + (paste_x_end - paste_x_start)
        
        img[bar_y_offset:bar_y_offset+bar_height, paste_x_start:paste_x_end] = bar_resized[:, bar_crop_start:bar_crop_end]
        
        mean_pixel_x = int((mean + 1) / 2 * click_width + click_min_x)
        cv2.line(img, (mean_pixel_x, 0), (mean_pixel_x, h), (255, 255, 255), 1)

    return img

def draw_click_indicator(img, click_val):
    """Draws a vertical bar indicating click probability/value."""
    h, w, _ = img.shape
    
    bar_w = 5
    bar_h = 40
    x = w - bar_w
    y_center = h // 2
    
    cv2.rectangle(img, (x, y_center - bar_h//2), (x + bar_w, y_center + bar_h//2), (50, 50, 50), -1)
    
    val = np.tanh(click_val)
    bar_fill_h = int(abs(val) * (bar_h / 2))
    
    if val > 0:
        cv2.rectangle(img, (x, y_center - bar_fill_h), (x + bar_w, y_center), (0, 255, 0), -1)
    else:
        cv2.rectangle(img, (x, y_center), (x + bar_w, y_center + bar_fill_h), (0, 0, 255), -1)
        
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=str, default=":99", help="The sandbox display ID")
    parser.add_argument("--model-name", type=str, default="ppo_fitcats_distributed", help="Name of the model to load")
    args = parser.parse_args()
    
    print(f"=== Saliency Map Visualization Tool ===")
    
    display_size = (420, 420)
    cv2.namedWindow("Saliency Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Saliency Map", display_size[0], display_size[1])
    cv2.waitKey(1)
    
    os.environ["DISPLAY"] = args.sandbox
    import pyautogui

    model_path = f"models/{args.model_name}.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    model.policy.train() 

    with open("calibration_data.json", "r") as f:
        calib = json.load(f)
        
    # Load templates and setup mask
    known_colors, not_cat_colors = load_templates("./cat_templates")
    agent_view_roi = calib["agent_view_roi"]
    poly_points = calib["playable_polygon"]
    cat_mask = np.zeros((agent_view_roi['h'], agent_view_roi['w']), dtype=np.uint8)
    rel_poly_points = [(px - agent_view_roi['x'], py - agent_view_roi['y']) for px, py in poly_points]
    pts = np.array(rel_poly_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(cat_mask, [pts], 255)

    # Frame stacks for each observation key
    frame_stack = {
        "board": deque(maxlen=4),
        "next_cat_type": deque(maxlen=4),
        "time_since_click": deque(maxlen=4),
        "cat_census": deque(maxlen=4),
        "pile_height": deque(maxlen=4)
    }

    with mss.mss() as sct:
        game_region = start_game(sct, calib, pyautogui)
        
        print("\nContinuously generating saliency maps. Press Ctrl+C to quit.")
        
        last_click_time = time.time()

        while True:
            try:
                game_img = np.array(sct.grab(game_region))
                game_img_bgr = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)
                
                # Preprocess current frame
                board_t, next_cat_t, census_t, pile_h_t = preprocess_observation(game_img_bgr, calib, known_colors, not_cat_colors, cat_mask)
                
                # Calculate time delta (dummy for visualization, or track real clicks?)
                # For visualization, we can just use 0 or a random value, or track clicks if we were playing.
                # Since this is just watching, let's use a dummy value or try to detect clicks?
                # Let's just use 1.0 to simulate "ready".
                time_delta_t = torch.tensor([1.0], dtype=torch.float32)
                
                # Update stacks
                if len(frame_stack["board"]) == 0:
                    for _ in range(4):
                        frame_stack["board"].append(board_t)
                        frame_stack["next_cat_type"].append(next_cat_t)
                        frame_stack["time_since_click"].append(time_delta_t)
                        frame_stack["cat_census"].append(census_t)
                        frame_stack["pile_height"].append(pile_h_t)
                else:
                    frame_stack["board"].append(board_t)
                    frame_stack["next_cat_type"].append(next_cat_t)
                    frame_stack["time_since_click"].append(time_delta_t)
                    frame_stack["cat_census"].append(census_t)
                    frame_stack["pile_height"].append(pile_h_t)
                
                # Stack tensors
                obs_dict = {
                    "board": torch.cat(list(frame_stack["board"]), dim=0),
                    "next_cat_type": torch.cat(list(frame_stack["next_cat_type"]), dim=0),
                    "time_since_click": torch.cat(list(frame_stack["time_since_click"]), dim=0),
                    "cat_census": torch.cat(list(frame_stack["cat_census"]), dim=0),
                    "pile_height": torch.cat(list(frame_stack["pile_height"]), dim=0)
                }
                
                saliency_heatmap = generate_saliency_map(model, obs_dict)
                mean_x, mean_click, std = get_policy_distribution(model, obs_dict)
                
                agent_view_roi = calib["agent_view_roi"]
                play_area_img = game_img_bgr[agent_view_roi['y']:agent_view_roi['y']+agent_view_roi['h'], agent_view_roi['x']:agent_view_roi['x']+agent_view_roi['w']]
                
                # --- KEY FIX: Convert to Grayscale for Visualization ---
                play_area_gray = cv2.cvtColor(play_area_img, cv2.COLOR_BGR2GRAY)
                play_area_gray_bgr = cv2.cvtColor(play_area_gray, cv2.COLOR_GRAY2BGR)
                
                play_area_resized = cv2.resize(play_area_gray_bgr, (saliency_heatmap.shape[1], saliency_heatmap.shape[0]))

                overlay = cv2.addWeighted(play_area_resized, 0.6, saliency_heatmap, 0.4, 0)
                
                overlay = draw_policy_heatmap_bar(overlay, mean_x, std, calib)
                overlay = draw_click_indicator(overlay, mean_click)
                
                display_overlay = cv2.resize(overlay, display_size, interpolation=cv2.INTER_NEAREST)
                
                cv2.putText(display_overlay, f"Mean: {mean_x:.2f}", (10, display_size[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_overlay, f"Mean: {mean_x:.2f}", (10, display_size[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.putText(display_overlay, f"Std: {std:.2f}", (10, display_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_overlay, f"Std: {std:.2f}", (10, display_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.imshow("Saliency Map", display_overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
