# FitCats RL Agent

This project uses deep reinforcement learning to train an AI agent to play the "Fit Cats" puzzle game. The agent learns to stack cats of the same size to merge them, aiming to achieve the highest possible score before the container overflows.

It uses `stable-baselines3` for the PPO algorithm and computer vision techniques (`OpenCV`, `mss`, `pyautogui`) to see the screen and interact with the game.

## Features

- **Sandboxed Environment**: Uses **Xephyr** to create isolated virtual displays, allowing you to use your computer while the agents train in the background.
- **Distributed Training**: Capable of running multiple game instances in parallel to train a single, shared model, significantly speeding up data collection and learning.
- **Custom Template-Based OCR**: A highly accurate, custom-built OCR system for reading the game score, replacing the less reliable general-purpose Tesseract.
- **Automated Setup & Calibration**: Scripts to automate the process of finding game coordinates and creating visual templates.

## Setup

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install System Dependencies**:
    ```bash
    sudo apt-get update
    sudo apt-get install xserver-xephyr fluxbox tesseract-ocr tesseract-ocr-all
    ```

## How to Use

The project is designed with a clear, step-by-step workflow.

### Step 1: Calibration (One-Time Setup)

First, you need to run the calibration script. This will launch a sandbox window and guide you through capturing the necessary screen coordinates and base templates for the game. **You only need to do this once.**

```bash
chmod +x calibrate_sandbox.sh
./calibrate_sandbox.sh
```
Follow the interactive prompts in the terminal.

### Step 2: Create Digit Templates

Next, build the custom OCR library. This script will launch a sandbox and ask you to identify digits as you play the game, creating a robust recognition library.

```bash
chmod +x create_digit_templates.sh
./create_digit_templates.sh
```
Follow the prompts. The more examples you provide for each digit, the more accurate the score reading will be.

### Step 3: Train the Agent

Once calibration and digit templates are complete, you can start training the agent.

To run the distributed training with **4 parallel agents**, use the following command:

```bash
python train_distributed.py --num-agents 4
```

This will launch 4 sandbox windows and begin training a single shared model. You can adjust `--num-agents` based on your CPU/RAM capacity.

### Step 4: Monitor Training

You can monitor the agent's learning progress using TensorBoard.

```bash
tensorboard --logdir logs/distributed
```
Navigate to `http://localhost:6006/` in your browser to view graphs of the agent's reward, score, and other metrics.

## Scripts Overview

- `train_distributed.py`: The main script for launching and managing distributed training.
- `fit_cats_env.py`: The custom OpenAI Gym/Gymnasium environment that defines the game logic, observations, and rewards.
- `setup_agent.py`: A comprehensive, interactive tool to calibrate all game coordinates.
- `create_digit_templates.py`: An interactive tool to build the custom OCR template library.
- `*.sh` scripts: Convenience wrappers for launching the Python scripts in the correct sandboxed environment.
