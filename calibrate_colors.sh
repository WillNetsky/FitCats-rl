#!/bin/bash

echo "--- Cat Color Calibration Sandbox ---"

# --- Launch Sandbox ---
DISP_ID=":99"
echo "Launching sandbox on $DISP_ID..."

Xephyr $DISP_ID -ac -screen 1355x1200 -title "Color Calibration" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

USER_DATA_DIR="$HOME/chrome-rl-color-calib"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser in Sandbox..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 8

# --- Run the Calibration Script ---
echo ""
echo "======================================================="
echo "Sandbox ready. The color calibration tool will now start."
echo "The debug window will appear on your MAIN screen."
echo "Play the game in the sandbox window."
echo "Click on cats in the 'Color Picker' window to save their colors."
echo "======================================================="
echo ""

# Run python on the CURRENT display (usually :0), passing the sandbox ID
/home/netsky/Code/FitCats-rl/.venv/bin/python calibrate_colors.py --sandbox $DISP_ID

# --- Cleanup ---
echo "Session finished. Closing sandbox."
kill $XEPHYR_PID
