#!/bin/bash

echo "--- Installing System Dependencies for FitCats-RL ---"

# Update package list
sudo apt-get update

# Install X11 utilities and Window Manager
echo "Installing Xephyr, Fluxbox, and Screen tools..."
sudo apt-get install -y xserver-xephyr fluxbox scrot python3-tk python3-dev

# Install Chromium (if not using Snap)
# Note: On Ubuntu, Chromium is usually a Snap, which is pre-installed or installed via snap.
# If you need a specific version or deb, add it here.
if ! command -v chromium &> /dev/null; then
    echo "Chromium not found. Installing via Snap..."
    sudo snap install chromium
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "--- Setup Complete ---"
echo "You can now run ./calibrate_sandbox.sh to start."
