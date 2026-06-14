#!/bin/bash

echo "--- Automated Game Control Speed Test ---"

# --- Launch Sandbox ---
DISP_ID=":99"
echo "Launching test sandbox on $DISP_ID..."

Xephyr $DISP_ID -ac -screen 1355x1200 -title "Control Test Sandbox" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

USER_DATA_DIR="$HOME/chrome-rl-control-test"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 8

# --- Run the Automated Test Script ---
echo ""
echo "======================================================="
echo "Sandbox ready. The automated test will now begin."
echo "The script will find the optimal click delay by itself."
echo "======================================================="
echo ""

DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python test_controls.py

# --- Cleanup ---
echo "Test finished. Closing sandbox."
kill $XEPHYR_PID
