#!/bin/bash

DISP_ID=":99"

echo "--- Launching Calibration Sandbox on $DISP_ID ---"

# Start Xephyr and redirect its output to /dev/null
Xephyr $DISP_ID -ac -screen 1355x1200 -title "Calibration Sandbox" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

# Start Fluxbox if available
if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

# Start Browser
USER_DATA_DIR="$HOME/chrome-rl-calib"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 8

echo ""
echo "======================================================="
echo "Sandbox Ready for Calibration!"
echo "Follow the prompts in this terminal to define ROIs."
echo "======================================================="
echo ""

# Run the setup tool on the main display, but tell it to capture from the sandbox
/home/netsky/Code/FitCats-rl/.venv/bin/python setup_agent.py --sandbox $DISP_ID

# Cleanup
echo "Calibration finished. Closing sandbox."
kill $XEPHYR_PID
