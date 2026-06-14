# FitCats-RL: A Reinforcement Learning Agent for "Fit Cats"

This project trains a reinforcement learning agent to play the Flash game "Fit Cats" using `stable-baselines3`. It uses a multi-headed input (image, cat count, etc.) and a sophisticated reward shaping strategy to learn the game.

## Project Structure

- `train_distributed.py`: The main script for training the agent. Launches multiple sandboxed environments.
- `fit_cats_env.py`: The custom OpenAI Gym environment for the game.
- `saliency_tool.py`: A research-grade tool to visualize the agent's "thought process" with saliency maps.
- `calibration_data.json`: Stores the screen coordinates for game elements.
- `highscores.json`: Tracks global and model-specific high scores.
- `models/`: Stores the trained agent models (`.zip`).
- `logs/`: Stores TensorBoard logs for each model.
- `highscores/`: Stores screenshots of high-score moments.
- `*.sh`: Utility scripts for setup, calibration, and debugging.

## Setup and Installation

This project is designed for a Linux environment with X11.

### 1. System Setup

First, run the system setup script to install all necessary system packages (like `Xephyr`, `fluxbox`, `chromium`) and Python dependencies.

```bash
chmod +x setup_system.sh
./setup_system.sh
```

### 2. Calibration

Before training, you must calibrate the agent to find the game on your screen. Run the calibration tool and follow the on-screen prompts in the terminal.

```bash
chmod +x calibrate_sandbox.sh
./calibrate_sandbox.sh
```
This will generate the `calibration_data.json` file.

## Usage

### Training a New Model

To train a new model from scratch, provide a unique `--model-name`.

```bash
python train_distributed.py --num-agents 4 --model-name my_first_model
```
This will create:
- `models/my_first_model.zip`
- `logs/my_first_model/` (for TensorBoard)
- `highscores_my_first_model.json`

### Resuming Training

To resume training for an existing model, simply run the command with the same model name.

```bash
python train_distributed.py --num-agents 4 --model-name my_first_model
```

### Monitoring

Use TensorBoard to monitor training progress.

```bash
tensorboard --logdir logs
```
Navigate to `http://localhost:6006/` in your browser. You will find:
- **Scalars:** Reward, loss, click ratio, etc.
- **Images:** Screenshots from the agent's perspective with overlaid stats.
- **Histograms:** Distribution of weights and biases for each layer.
- **Graphs:** The computation graph of the policy network.

### Visualizing the Agent's Brain (Saliency Maps)

Use the saliency tool to see what the agent is "looking at" in real-time.

```bash
chmod +x saliency_tool.sh
./saliency_tool.sh --model-name my_first_model
```
This will open a window showing:
- **Saliency Heatmap:** Red/yellow areas show what pixels are most influential.
- **Policy Heatmap Bar:** A Red/Yellow/Green bar showing the probability of clicking at different horizontal positions.
- **Click Indicator:** A vertical bar showing the agent's confidence in clicking vs. waiting.
