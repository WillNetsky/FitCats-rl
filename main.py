from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from fit_cats_env import FitCatsEnv
import os
import time

def main():
    # --- Setup ---
    log_dir = "logs/"
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    # --- Environment ---
    env = FitCatsEnv()

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=os.path.join(model_dir, run_id),
        name_prefix="fit_cats_ppo"
    )

    # --- Model Loading/Creation ---
    # Path to the model you want to resume from
    model_path = os.path.join(model_dir, "fit_cats_ppo_final.zip")

    if os.path.exists(model_path):
        print(f"--- Resuming training from {model_path} ---")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print("--- Starting new training ---")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir,
            n_steps=256
        )

    # --- Training ---
    print("\nMake sure the game window is visible and unobstructed.")
    print(f"TensorBoard logs will be saved in: {log_dir}")
    print(f"Model checkpoints will be saved in: {model_dir}{run_id}/")
    print(f"\nTo view progress, run: tensorboard --logdir {log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=50000, 
            callback=checkpoint_callback,
            tb_log_name=f"PPO_{run_id}",
            reset_num_timesteps=False  # Set to False to continue step count from loaded model
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Save final model
        model.save(model_path) # Overwrite the final model
        env.close()
        print(f"\nTraining complete. Final model saved to {model_path}")

if __name__ == '__main__':
    main()
