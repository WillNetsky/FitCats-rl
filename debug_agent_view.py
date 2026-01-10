from fit_cats_env import FitCatsEnv
import cv2

def main():
    print("This script will initialize the environment and save one")
    print("84x84 image representing the agent's view.")
    
    # Initialize the environment (this will trigger calibration)
    env = FitCatsEnv()
    
    # Get the initial observation from the reset method
    obs, _ = env.reset()
    
    # The observation is already an 84x84 numpy array.
    # We just need to save it.
    # Note: OpenCV expects BGR, but the observation is already in that format
    # because of how the environment processes it.
    cv2.imwrite("agent_view.png", obs)
    
    print("\nSaved 'agent_view.png'.")
    print("This is the 84x84 pixel image the agent uses as input.")
    
    env.close()

if __name__ == "__main__":
    main()
