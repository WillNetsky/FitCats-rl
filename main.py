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

    # --- Model ---
    # We set n_steps to a small value to force frequent logging.
    # The default is 2048, which can take a while to complete.
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        n_steps=256  # Log data much more frequently
    )

    # --- Training ---
    print("\n--- Starting Training ---")
    print("Make sure the game window is visible and unobstructed.")
    print(f"TensorBoard logs will be saved in: {log_dir}")
    print(f"Model checkpoints will be saved in: {model_dir}{run_id}/")
    print(f"\nTo view progress, run: tensorboard --logdir {log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=50000, 
            callback=checkpoint_callback,
            tb_log_name=f"PPO_{run_id}"
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Save final model
        final_model_path = os.path.join(model_dir, "fit_cats_ppo_final.zip")
        model.save(final_model_path)
        env.close()
        print(f"\nTraining complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()
