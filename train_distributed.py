import os
import subprocess
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import argparse
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # The Monitor wrapper now handles the rollout stats (ep_rew_mean, etc.)
        # We just need to log our custom game stats from the info dict.
        for info in self.locals.get("infos", []):
            if "game/next_cat_size" in info:
                self.logger.record("game/next_cat_size", info["game/next_cat_size"])
            if "game/did_click" in info:
                self.logger.record("game/click_rate", info["game/did_click"])
            
            # The Monitor wrapper puts the final episode info in a special key
            if "episode" in info:
                self.logger.record("game/final_score", info["episode"]["r"])
                self.logger.record("game/ep_length", info["episode"]["l"])
        return True

def make_env(display_id, rank):
    """
    Utility function for multiprocessed env.
    :param display_id: (str) the X display to use
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Set the DISPLAY variable *before* importing the env
        os.environ["DISPLAY"] = display_id
        from fit_cats_env import FitCatsEnv
        
        # Wrap the env in a Monitor to log rollout stats
        log_file = os.path.join("logs/distributed/", f"monitor_{rank}.csv")
        env = Monitor(FitCatsEnv(), log_file)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=2, help="Number of parallel agents to run")
    args = parser.parse_args()

    num_agents = args.num_agents
    print(f"--- Launching {num_agents} agents for distributed training ---")

    xephyr_pids = []
    
    # --- Launch Xephyr and Browser Instances ---
    for i in range(num_agents):
        instance_id = i + 1
        display_id = f":9{instance_id}"
        
        print(f"Starting Xephyr for Instance {instance_id} on {display_id}...")
        xephyr_cmd = ["Xephyr", display_id, "-ac", "-screen", "1355x1200", "-title", f"FitCats Agent {instance_id}"]
        proc = subprocess.Popen(xephyr_cmd)
        xephyr_pids.append(proc)
        time.sleep(2)

        print(f"Starting Fluxbox for Instance {instance_id}...")
        fluxbox_cmd = ["fluxbox"]
        subprocess.Popen(fluxbox_cmd, env={**os.environ, "DISPLAY": display_id})
        time.sleep(1)

        print(f"Starting Browser for Instance {instance_id}...")
        user_data_dir = f"{os.path.expanduser('~')}/chrome-rl-instance-{instance_id}"
        os.makedirs(user_data_dir, exist_ok=True)
        
        browser_cmd = [
            "/snap/bin/chromium",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--start-maximized",
            "--no-sandbox",
            "https://www.newgrounds.com/portal/view/913713"
        ]
        subprocess.Popen(browser_cmd, env={**os.environ, "DISPLAY": display_id})
        
    print("\nAll instances launched. Waiting 10 seconds for browsers to load...")
    time.sleep(10)

    # --- Create the Vectorized Environment ---
    env_fns = [make_env(f":9{i+1}", i) for i in range(num_agents)]
    env = SubprocVecEnv(env_fns)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    # --- Model and Training ---
    model_path = "models/ppo_fitcats_distributed.zip"
    log_dir = "logs/distributed/"
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model")
    tensorboard_callback = TensorboardCallback()

    if os.path.exists(model_path):
        print("--- Resuming training from saved model ---")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print("--- Starting new training ---")
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=256)

    try:
        model.learn(total_timesteps=200000, callback=[checkpoint_callback, tensorboard_callback])
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Saving final model...")
        model.save(model_path)
        env.close()
        
        # Cleanup Xephyr processes
        for proc in xephyr_pids:
            proc.terminate()
        print("Training complete.")

if __name__ == '__main__':
    if 'fork' in os.get_exec_path():
        os.set_start_method('forkserver', force=True)
    main()
