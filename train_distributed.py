import os
import subprocess
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import Image, TensorBoardOutputFormat
import argparse
import numpy as np
import cv2
import torch
from collections import deque
import datetime
import json

# --- Constants ---
MAX_CAT_TYPES = 15
IMG_SIZE = 160 

class MissionControlCallback(BaseCallback):
    """
    A custom callback for a terminal dashboard and periodic TensorBoard snapshots with stats.
    """
    def __init__(self, num_agents, total_timesteps, n_steps, model_name, log_freq=256, verbose=0):
        super(MissionControlCallback, self).__init__(verbose)
        self.num_agents = num_agents
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.model_name = model_name
        self.log_freq = log_freq
        
        self.agent_current_scores = [0] * num_agents
        self.agent_last_final_scores = [0] * num_agents
        self.agent_session_high_scores = [0] * num_agents
        self.agent_steps = [0] * num_agents
        self.agent_cat_counts = [0] * num_agents
        self.agent_last_action_x = [0.0] * num_agents
        self.agent_did_click = [0] * num_agents
        self.agent_max_cat_types = [0] * num_agents

        self.last_update_time = time.time()
        self.last_update_duration = 0
        self.graph_logged = False
        
        self.click_history = deque(maxlen=1000)
        self.cat_count_history = deque(maxlen=1000)
        self.pile_height_history = deque(maxlen=1000)
        self.successful_click_history = deque(maxlen=1000)
        self.total_click_history = deque(maxlen=1000)
        self.attempted_click_history = deque(maxlen=1000)
        
        self.final_score_history = deque(maxlen=100)
        self.final_cat_count_history = deque(maxlen=100)
        self.final_max_cat_history = deque(maxlen=100)
        self.final_pile_height_history = deque(maxlen=100)
        self.final_clicks_history = deque(maxlen=100)

    def _get_highscores(self):
        """Reads global and model-specific highscores."""
        global_high = 0
        global_max_cats = 0
        model_high = 0
        model_max_cats = 0
        
        try:
            if os.path.exists("highscores.json"):
                with open("highscores.json", "r") as f:
                    data = json.load(f)
                    global_high = data.get("global_high_score", 0)
                    global_max_cats = data.get("global_max_cats", 0)
            
            model_file = f"highscores_{self.model_name}.json"
            if os.path.exists(model_file):
                with open(model_file, "r") as f:
                    data = json.load(f)
                    model_high = data.get("high_score", 0)
                    model_max_cats = data.get("max_cats", 0)
        except: pass
        
        return global_high, global_max_cats, model_high, model_max_cats

    def _on_step(self) -> bool:
        # --- Log Computation Graph (Disabled for RecurrentPPO) ---
        
        now = time.time()
        self.last_update_duration = now - self.last_update_time
        self.last_update_time = now
        self.logger.record("time/update_duration_seconds", self.last_update_duration)

        total_did_click = 0
        total_attempted_clicks = 0
        total_successful_clicks = 0
        # --- KEY FIX: Initialize missing variables ---
        total_cats_on_board = 0
        max_cat_type_seen = 0
        
        for i in range(self.num_agents):
            info = self.locals["infos"][i]
            self.agent_current_scores[i] = info.get("game/score", self.agent_current_scores[i])
            self.agent_steps[i] = info.get("game/step_count", self.agent_steps[i])
            self.agent_cat_counts[i] = info.get("game/cat_count", self.agent_cat_counts[i])
            self.agent_last_action_x[i] = info.get("game/action_x", self.agent_last_action_x[i])
            self.agent_max_cat_types[i] = info.get("game/max_cat_type", self.agent_max_cat_types[i])
            
            did_click = info.get("game/did_click", 0)
            self.agent_did_click[i] = did_click
            total_did_click += did_click
            
            attempted_click = info.get("game/attempted_click", 0)
            total_attempted_clicks += attempted_click
            
            successful_click = info.get("game/successful_click", 0)
            total_successful_clicks += successful_click
            
            # --- KEY FIX: Accumulate stats ---
            total_cats_on_board += self.agent_cat_counts[i]
            if self.agent_max_cat_types[i] > max_cat_type_seen:
                max_cat_type_seen = self.agent_max_cat_types[i]

            if "episode" in info:
                final_score = info.get("game/final_score", 0)
                final_cats = info.get("game/final_cat_count", 0)
                final_max_cat = info.get("game/final_max_cat_type", 0)
                final_pile = info.get("game/final_pile_height", 0)
                final_clicks = info.get("game/final_clicks", 0)
                
                self.agent_last_final_scores[i] = final_score
                if final_score > self.agent_session_high_scores[i]:
                    self.agent_session_high_scores[i] = final_score
                
                self.final_score_history.append(final_score)
                self.final_cat_count_history.append(final_cats)
                self.final_max_cat_history.append(final_max_cat)
                self.final_pile_height_history.append(final_pile)
                self.final_clicks_history.append(final_clicks)
                
                print(f"[Episode End] Agent {i+1} | Score: {final_score} | Cats: {final_cats:.0f} | Max Cat: {final_max_cat} | Pile: {final_pile:.2f}")
                
                self.agent_current_scores[i] = 0
                self.agent_steps[i] = 0
                self.agent_cat_counts[i] = 0
                self.agent_last_action_x[i] = 0.0
                self.agent_did_click[i] = 0
                self.agent_max_cat_types[i] = 0

        self.click_history.append(total_did_click)
        if len(self.click_history) > 0:
            avg_clicks_per_step = sum(self.click_history) / len(self.click_history)
            click_ratio = avg_clicks_per_step / self.num_agents
        else:
            click_ratio = 0.0
        self.logger.record("game/executed_click_ratio", click_ratio)
        
        self.attempted_click_history.append(total_attempted_clicks)
        if len(self.attempted_click_history) > 0:
            avg_attempted_clicks = sum(self.attempted_click_history) / len(self.attempted_click_history)
            attempted_click_ratio = avg_attempted_clicks / self.num_agents
        else:
            attempted_click_ratio = 0.0
        self.logger.record("game/attempted_click_ratio", attempted_click_ratio)
        
        self.successful_click_history.append(total_successful_clicks)
        self.total_click_history.append(total_did_click)
        
        total_clicks_in_history = sum(self.total_click_history)
        if total_clicks_in_history > 0:
            score_increase_rate = sum(self.successful_click_history) / total_clicks_in_history
        else:
            score_increase_rate = 0.0
        self.logger.record("game/score_increase_rate", score_increase_rate)
        
        self.cat_count_history.append(total_cats_on_board)
        if len(self.cat_count_history) > 0:
            avg_cats = sum(self.cat_count_history) / (len(self.cat_count_history) * self.num_agents)
        else:
            avg_cats = 0.0
            
        self.logger.record("game/avg_cats_on_board", avg_cats)
        self.logger.record("game/max_cat_type_seen", max_cat_type_seen)

        current_pile_heights = self.locals["new_obs"]["pile_height"][:, -1]
        self.pile_height_history.append(np.mean(current_pile_heights))
        if len(self.pile_height_history) > 0:
            avg_pile_height = np.mean(self.pile_height_history)
        else:
            avg_pile_height = 0.0
        self.logger.record("game/avg_pile_height", avg_pile_height)
        
        if len(self.final_score_history) > 0:
            self.logger.record("game/avg_final_score", np.mean(self.final_score_history))
            self.logger.record("game/avg_final_cat_count", np.mean(self.final_cat_count_history))
            self.logger.record("game/avg_final_max_cat_type", np.mean(self.final_max_cat_history))
            self.logger.record("game/avg_final_pile_height", np.mean(self.final_pile_height_history))
            self.logger.record("game/avg_final_clicks", np.mean(self.final_clicks_history))

        # --- Periodic Logging (Scalars/Images) ---
        if self.n_calls % self.log_freq == 0:
            fps = self.num_agents / (self.last_update_duration + 1e-6)
            
            steps_until_update = self.n_steps - (self.n_calls % self.n_steps)
            time_to_update = steps_until_update * self.last_update_duration
            time_to_update_str = f"{time_to_update:.1f}s"
            
            total_gameplay_seconds = self.num_timesteps
            gameplay_time_str = str(datetime.timedelta(seconds=int(total_gameplay_seconds)))

            try:
                current_entropy = 0.0 
            except Exception as e:
                current_entropy = 0.0

            print(f"\n--- Update #{self.n_calls} | FPS: {fps:.1f} | Weight Update In: {time_to_update_str} | Total Gameplay: {gameplay_time_str} ---")
            scores_str = ", ".join([f"{s:<4}" for s in self.agent_current_scores])
            cats_str = ", ".join([f"{c:<2.0f}" for c in self.agent_cat_counts])
            print(f"Scores: [{scores_str}]")
            print(f"Cats  : [{cats_str}]")
            print(f"Exec Click: {click_ratio:.2f} | Attempt Click: {attempted_click_ratio:.2f} | Score Inc Rate: {score_increase_rate:.2f}")

            # Log Images
            try:
                g_high, g_max_cats, m_high, m_max_cats = self._get_highscores()
                
                images = self.training_env.get_attr('last_obs_img')
                for i in range(self.num_agents):
                    if images[i] is not None:
                        img_with_stats = images[i].copy()
                        h, w, _ = img_with_stats.shape
                        
                        if self.agent_did_click[i] > 0:
                            action_str = f"CLICK ({self.agent_last_action_x[i]:.2f})"
                        else:
                            action_str = "WAIT"

                        stats = [
                            f"Score: {self.agent_current_scores[i]}",
                            f"Cats: {self.agent_cat_counts[i]:.0f}",
                            f"Last Final: {self.agent_last_final_scores[i]}",
                            f"Session High: {self.agent_session_high_scores[i]}",
                            f"Model High: {m_high} (Max Cats: {m_max_cats})",
                            f"Global High: {g_high} (Max Cats: {g_max_cats})",
                            f"Action: {action_str}",
                        ]
                        
                        y0 = h - 20
                        dy = 35
                        font_scale = 1.0
                        text_color = (255, 255, 0) # Cyan in BGR
                        
                        for j, line in enumerate(reversed(stats)):
                            y = y0 - j * dy
                            cv2.putText(img_with_stats, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4)
                            cv2.putText(img_with_stats, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

                        pile_height_norm = self.locals["new_obs"]["pile_height"][i, -1]
                        line_y = int((1.0 - pile_height_norm) * h)
                        cv2.line(img_with_stats, (0, line_y), (w, line_y), (0, 0, 255), 2)

                        img_rgb = cv2.cvtColor(img_with_stats, cv2.COLOR_BGR2RGB)
                        self.logger.record(f"agent_view/agent_{i}", Image(img_rgb, "HWC"), exclude=("stdout", "log", "json", "csv"))
            except Exception as e:
                print(f"\nError logging images: {e}")
            
            self.logger.dump(self.num_timesteps)

        # --- Log Parameter Histograms (Every Weight Update) ---
        if self.n_calls % self.n_steps == 0:
            print(f"\n[System] --- Weights Updated at Step {self.num_timesteps} ---")
            
            try:
                tb_writer = None
                for format in self.logger.output_formats:
                    if isinstance(format, TensorBoardOutputFormat):
                        tb_writer = format.writer
                        break
                
                if tb_writer:
                    for name, param in self.model.policy.named_parameters():
                        tb_writer.add_histogram(f"params/{name}", param, self.num_timesteps)
                else:
                    print("Warning: No TensorBoard writer found for histograms.")
            except Exception as e:
                print(f"\nError logging histograms: {e}")

        return True

    def _on_training_end(self) -> None:
        print()

def make_env(display_id, rank, log_dir):
    """Utility function for multiprocessed env."""
    def _init():
        os.environ["DISPLAY"] = display_id
        from fit_cats_env import FitCatsEnv
        log_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        env = Monitor(FitCatsEnv(), log_file)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=2, help="Number of parallel agents to run")
    default_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--model-name", type=str, default=default_name, help="Name of the model (for saving/loading)")
    args = parser.parse_args()

    num_agents = args.num_agents
    model_name = args.model_name
    n_steps = 1024
    total_timesteps = 200000
    
    os.environ["FITCATS_MODEL_NAME"] = model_name
    
    print(f"--- Launching {num_agents} agents for model '{model_name}' ---")

    xephyr_pids = []
    dev_null = open(os.devnull, 'w')

    for i in range(num_agents):
        instance_id = i + 1
        display_id = f":9{instance_id}"
        print(f"Starting Xephyr for Instance {instance_id} on {display_id}...")
        xephyr_cmd = ["Xephyr", display_id, "-ac", "-screen", "1355x1200", "-title", f"FitCats Agent {instance_id}"]
        proc = subprocess.Popen(xephyr_cmd, stdout=dev_null, stderr=dev_null)
        xephyr_pids.append(proc)
        time.sleep(2)
        subprocess.Popen(["fluxbox"], env={**os.environ, "DISPLAY": display_id}, stdout=dev_null, stderr=dev_null)
        time.sleep(1)
        user_data_dir = f"{os.path.expanduser('~')}/chrome-rl-instance-{instance_id}"
        os.makedirs(user_data_dir, exist_ok=True)
        browser_cmd = ["/snap/bin/chromium", f"--user-data-dir={user_data_dir}", "--no-first-run", "--start-maximized", "--no-sandbox", "https://www.newgrounds.com/portal/view/913713"]
        subprocess.Popen(browser_cmd, env={**os.environ, "DISPLAY": display_id}, stdout=dev_null, stderr=dev_null)
        
    print("\nAll instances launched. Waiting 10 seconds for browsers to load...")
    time.sleep(10)

    model_path = f"models/{model_name}.zip"
    log_dir = f"logs/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)

    env_fns = [make_env(f":9{i+1}", i, log_dir) for i in range(num_agents)]
    env = SubprocVecEnv(env_fns)
    
    env = VecFrameStack(env, n_stack=4)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model")
    mission_control_callback = MissionControlCallback(num_agents=num_agents, total_timesteps=total_timesteps, n_steps=n_steps, model_name=model_name, log_freq=256)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(model_path):
        print(f"\n--- Resuming training from {model_path} ---")
        model = RecurrentPPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print(f"\n--- Starting new training for {model_name} ---")
        model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=0, tensorboard_log=log_dir, n_steps=n_steps)

    try:
        if not os.path.exists(model_path):
             model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=0, tensorboard_log=log_dir, n_steps=n_steps)
        
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, mission_control_callback], tb_log_name=run_id)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print(f"\nSaving final model to {model_path}...")
        model.save(model_path)
        env.close()
        for proc in xephyr_pids:
            proc.terminate()
        dev_null.close()
        print("Training complete.")

if __name__ == '__main__':
    if 'fork' in os.get_exec_path():
        os.set_start_method('forkserver', force=True)
    main()
