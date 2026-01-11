from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from fit_cats_env import FitCatsEnv
import os
import time
import numpy as np
import argparse

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard and saving a summary.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_scores = []

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            infos = self.locals["infos"]
            for info in infos:
                if "next_cat_size" in info:
                    self.logger.record("game/next_cat_size", info["next_cat_size"])
                if "did_click" in info:
                    self.logger.record("game/click_rate", info["did_click"])
                
                if "is_game_over" in info and info["is_game_over"]:
                    score = info["score"]
                    self.logger.record("game/final_score", score)
                    self.episode_scores.append(score)
        return True

    def save_summary(self, path):
        if not self.episode_scores:
            return
        
        with open(path, "w") as f:
            f.write("--- Fit Cats Training Summary ---\n")
            f.write(f"Total Episodes: {len(self.episode_scores)}\n")
            f.write(f"Best Score: {np.max(self.episode_scores)}\n")
            f.write(f"Average Score: {np.mean(self.episode_scores):.2f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Last 20 Scores: {self.episode_scores[-20:]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="1", help="Unique ID for this training run")
    args = parser.parse_args()

    # --- Setup Unique Paths ---
    run_name = f"fit_cats_instance_{args.id}"
    log_dir = f"logs/{run_name}"
    model_dir = f"models/{run_name}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"--- Starting Instance {args.id} ---")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # --- Environment ---
    env = FitCatsEnv()
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="checkpoint"
    )
    tensorboard_callback = TensorboardCallback()

    # --- Model ---
    model_path = os.path.join(model_dir, "final_model.zip")

    # Hyperparameters
    learning_rate = 3e-4
    ent_coef = 0.01
    n_steps = 256

    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir,
                         learning_rate=learning_rate, ent_coef=ent_coef, n_steps=n_steps)
    else:
        print("Starting new model")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            n_steps=n_steps
        )

    try:
        model.learn(
            total_timesteps=100000, 
            callback=[checkpoint_callback, tensorboard_callback],
            tb_log_name=run_name,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        model.save(model_path)
        tensorboard_callback.save_summary(os.path.join(model_dir, "summary.txt"))
        env.close()
        print(f"Saved to {model_path}")

if __name__ == '__main__':
    main()
