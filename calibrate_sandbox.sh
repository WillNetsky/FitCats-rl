#!/bin/bash

DISP_ID=":99"

echo "--- Launching Calibration Sandbox on $DISP_ID ---"

# Start Xephyr
# Increased width to 1355 and height to 1200
Xephyr $DISP_ID -ac -screen 1355x1200 -title "Calibration Sandbox" &
XEPHYR_PID=$!
sleep 2

# Start Fluxbox
DISPLAY=$DISP_ID fluxbox &
sleep 1

# Start Browser
USER_DATA_DIR="$HOME/chrome-rl-calibration"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 5

echo ""
echo "======================================================="
echo "Sandbox Ready for Calibration!"
echo "Running setup_agent.py inside the sandbox..."
echo "======================================================="
echo ""

# Run the setup tool inside the sandbox
DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python setup_agent.py

# Cleanup
kill $XEPHYR_PID
