#!/bin/bash

DISP_ID=":99"

echo "--- Launching Digit Template Creation Sandbox on $DISP_ID ---"

# Start Xephyr with the standard resolution
Xephyr $DISP_ID -ac -screen 1355x1200 -title "Digit Template Creator" &
XEPHYR_PID=$!
sleep 2

# Start Fluxbox
DISPLAY=$DISP_ID fluxbox &
sleep 1

# Start Browser
USER_DATA_DIR="$HOME/chrome-rl-digit-creation"
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
echo "Sandbox Ready for Digit Template Creation!"
echo "Running create_digit_templates.py inside the sandbox..."
echo "Play the game in the sandbox window to get a score."
echo "Then, follow the prompts in the terminal where you ran this script."
echo "======================================================="
echo ""

# Run the digit template creation tool inside the sandbox
DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python create_digit_templates.py

# Cleanup
echo "Digit creation session finished. Closing sandbox."
kill $XEPHYR_PID
