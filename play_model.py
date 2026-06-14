import cv2
import numpy as np
import mss
import os
import json
import time
import argparse
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from collections import deque
from scipy.stats import norm
from utils import start_game
from fit_cats_env import FitCatsEnv, get_cat_census_and_next, load_templates, MAX_CAT_TYPES, IMG_SIZE, find_candidate_circles, get_dominant_color, filter_nested_circles

# NOTE: PyAutoGUI is imported inside main() after setting the DISPLAY variable.

def preprocess_observation(img, calib, known_colors, not_cat_colors, radius_ranges, cat_mask):
    """Prepares a raw image into the format the agent expects."""
    roi = calib["agent_view_roi"]
    board_img = img[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    board_obs = cv2.resize(board_img, (IMG_SIZE, IMG_SIZE))
    
    # Get additional features
    roi_next = calib["next_cat_roi"]
    next_cat_img = img[roi_next['y']:roi_next['y']+roi_next['h'], roi_next['x']:roi_next['x']+roi_next['w']]
    
    cat_census, next_cat_type, pile_height = get_cat_census_and_next(board_img, next_cat_img, cat_mask, known_colors, not_cat_colors, radius_ranges)
    
    # Convert to tensors
    board_tensor = torch.tensor(board_obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    next_cat_one_hot = torch.zeros(MAX_CAT_TYPES, dtype=torch.float32)
    if next_cat_type != -1 and next_cat_type < MAX_CAT_TYPES:
        next_cat_one_hot[next_cat_type] = 1.0
        
    cat_census_tensor = torch.tensor(cat_census, dtype=torch.float32)
    pile_height_tensor = torch.tensor([pile_height], dtype=torch.float32)
    
    return board_tensor, next_cat_one_hot, cat_census_tensor, pile_height_tensor

def get_policy_distribution(model, obs_dict, lstm_states, episode_starts):
    """Gets the probability distribution of the discrete actions using the full model."""
    device = model.device
    
    # Ensure inputs are on device and have batch dim
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray): v = torch.tensor(v)
        if v.dtype != torch.float32: v = v.float()
        if v.dim() == 1: obs_dict[k] = v.unsqueeze(0).to(device)
        elif v.dim() == 3: obs_dict[k] = v.unsqueeze(0).to(device)
        else: obs_dict[k] = v.to(device)

    if lstm_states is None:
        # If we don't have states yet, we can't easily run get_distribution without knowing shapes.
        # We will return uniform distribution as a fallback.
        return np.ones(IMG_SIZE)/IMG_SIZE, np.array([0.5, 0.5])
    
    # Convert numpy states to tensor
    # lstm_states is (h, c). Each is (n_layers, n_envs, hidden_size)
    if lstm_states is not None:
        lstm_states_tensor = (
            torch.tensor(lstm_states[0], device=device),
            torch.tensor(lstm_states[1], device=device)
        )

    episode_starts_tensor = torch.tensor(episode_starts, device=device)

    with torch.no_grad():
        # This runs the full forward pass: CNN -> LSTM -> MLP -> ActionNet
        # --- KEY FIX: Unpack tuple return ---
        dist, _ = model.policy.get_distribution(obs_dict, lstm_states_tensor, episode_starts_tensor)
        
        dists = dist.distribution
        x_dist = dists[0]
        click_dist = dists[1]
        
        x_probs = x_dist.probs.cpu().numpy()[0]
        click_probs = click_dist.probs.cpu().numpy()[0]
        
        return x_probs, click_probs

def generate_saliency_map_cnn_only(model, obs_dict):
    """Generates a saliency map based on CNN features only (ignoring LSTM/Policy)."""
    device = model.device
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray): v = torch.tensor(v)
        if v.dtype != torch.float32: v = v.float()
        if v.dim() == 1: obs_dict[k] = v.unsqueeze(0).to(device)
        elif v.dim() == 3: obs_dict[k] = v.unsqueeze(0).to(device)
        else: obs_dict[k] = v.to(device)
            
    obs_dict["board"].requires_grad_()
    
    # Just extract features. This uses the CNN.
    features = model.policy.extract_features(obs_dict)
    
    # We want to maximize the activation of the features
    # Simple heuristic: Maximize the mean activation
    target = features.mean()
    
    target.backward()
    
    saliency, _ = torch.max(obs_dict["board"].grad.data.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()
    
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    saliency_heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    
    return saliency_heatmap

def draw_policy_heatmap_bar(img, x_probs, calib):
    """Draws the discrete policy distribution as a colored bar."""
    h, w, _ = img.shape # 160x160
    
    orig_view_w = calib["agent_view_roi"]["w"]
    scale_factor = w / orig_view_w
    agent_view_x = calib["agent_view_roi"]["x"]
    
    orig_click_min_x = calib["click_x_min_rel"] - agent_view_x
    orig_click_max_x = calib["click_x_max_rel"] - agent_view_x
    
    click_min_x = int(orig_click_min_x * scale_factor)
    click_max_x = int(orig_click_max_x * scale_factor)
    click_width = click_max_x - click_min_x
    
    if click_width <= 0: return img

    # --- KEY FIX: Use absolute probability for color ---
    probs_norm = np.clip(x_probs / 0.1, 0, 1.0)
    
    bar_bgr = np.zeros((1, len(x_probs), 3), dtype=np.uint8)
    for i in range(len(x_probs)):
        val = probs_norm[i]
        r = int(val * 255)
        b = int((1.0 - val) * 255)
        g = 0
        bar_bgr[0, i] = (b, g, r)
            
    bar_height = 10
    bar_y_offset = 10
    
    bar_resized = cv2.resize(bar_bgr, (click_width, bar_height), interpolation=cv2.INTER_NEAREST)
    
    paste_x_start = max(0, click_min_x)
    paste_x_end = min(w, click_max_x)
    
    if paste_x_end > paste_x_start:
        bar_crop_start = paste_x_start - click_min_x
        bar_crop_end = bar_crop_start + (paste_x_end - paste_x_start)
        
        img[bar_y_offset:bar_y_offset+bar_height, paste_x_start:paste_x_end] = bar_resized[:, bar_crop_start:bar_crop_end]
        
        max_idx = np.argmax(x_probs)
        max_pixel_x = int((max_idx / (len(x_probs) - 1)) * click_width + click_min_x)
        cv2.line(img, (max_pixel_x, 0), (max_pixel_x, h), (255, 255, 255), 1)

    return img

def draw_click_indicator(img, click_probs):
    """Draws a vertical bar indicating click probability."""
    h, w, _ = img.shape
    
    bar_w = 10
    bar_h = 60
    x = w - bar_w
    y_center = h // 2
    
    cv2.rectangle(img, (x, y_center - bar_h//2), (x + bar_w, y_center + bar_h//2), (50, 50, 50), -1)
    
    click_p = click_probs[1]
    
    bar_fill_h = int(click_p * bar_h)
    
    cv2.rectangle(img, (x, y_center + bar_h//2 - bar_fill_h), (x + bar_w, y_center + bar_h//2), (0, 255, 0), -1)
    
    thresh_y = y_center
    cv2.line(img, (x, thresh_y), (x + bar_w, thresh_y), (255, 255, 255), 1)
        
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", type=str, default=":99", help="The sandbox display ID")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to load")
    args = parser.parse_args()
    
    print(f"=== Play Model: {args.model_name} ===")
    
    # Handle Display
    main_display = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = args.sandbox
    import pyautogui
    if main_display: os.environ["DISPLAY"] = main_display
    else: 
        if "DISPLAY" in os.environ: del os.environ["DISPLAY"]

    model_path = f"models/{args.model_name}.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model...")
    try:
        model = RecurrentPPO.load(model_path)
    except:
        print("Could not load as RecurrentPPO, trying PPO...")
        model = PPO.load(model_path)
    
    # Initialize Environment (connected to sandbox)
    os.environ["FITCATS_MODEL_NAME"] = args.model_name
    os.environ["DISPLAY"] = args.sandbox
    env = FitCatsEnv()
    if main_display: os.environ["DISPLAY"] = main_display
    
    # Load templates for visualization
    known_colors, not_cat_colors, radius_ranges = load_templates("./cat_templates")
    
    # Larger display size
    display_size = (840, 840)
    
    cv2.namedWindow("Saliency & Policy")
    cv2.moveWindow("Saliency & Policy", 0, 0)

    obs, _ = env.reset()
    
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    def make_env(): return env
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    obs = vec_env.reset()
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    print("\nPlaying... Press Ctrl+C to stop.")
    
    last_time = time.time()
    
    while True:
        try:
            # Predict action with state
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
            episode_starts = np.zeros((1,), dtype=bool) 
            
            # Step environment
            obs, reward, done, info = vec_env.step(action)
            if done[0]:
                episode_starts = np.ones((1,), dtype=bool)
                lstm_states = None 
            
            # --- Visualization ---
            
            # 1. Saliency & Policy
            obs_tensor = {}
            for k, v in obs.items():
                t = torch.tensor(v).to(model.device)
                if k == 'board':
                    t = t.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
                    t = t.float() / 255.0
                obs_tensor[k] = t

            saliency_map = generate_saliency_map_cnn_only(model, obs_tensor)
            
            # --- KEY FIX: Pass states to get_policy_distribution ---
            x_probs, click_probs = get_policy_distribution(model, obs_tensor, lstm_states, episode_starts)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            
            max_x_prob = np.max(x_probs)
            click_p = click_probs[1]
            print(f"\rFPS: {fps:.1f} | Max X Prob: {max_x_prob:.3f} | Click Prob: {click_p:.3f}   ", end="")
            
            # Get raw image from env for background
            raw_img = env.last_obs_img
            if raw_img is not None:
                agent_roi = env.calib["agent_view_roi"]
                play_area_img = raw_img[agent_roi['y']:agent_roi['y']+agent_roi['h'], agent_roi['x']:agent_roi['x']+agent_roi['w']]
                
                # Convert to Grayscale for Visualization
                play_area_gray = cv2.cvtColor(play_area_img, cv2.COLOR_BGR2GRAY)
                play_area_gray_bgr = cv2.cvtColor(play_area_gray, cv2.COLOR_GRAY2BGR)
                
                play_area_resized = cv2.resize(play_area_gray_bgr, (saliency_map.shape[1], saliency_map.shape[0]))
                
                overlay = cv2.addWeighted(play_area_resized, 0.6, saliency_map, 0.4, 0)
                
                overlay = draw_policy_heatmap_bar(overlay, x_probs, env.calib)
                overlay = draw_click_indicator(overlay, click_probs)
                
                display_overlay = cv2.resize(overlay, display_size, interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Saliency & Policy", display_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
