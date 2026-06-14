#!/bin/bash

echo "--- Play Model Sandbox ---"

# --- Launch Sandbox ---
DISP_ID=":99"
echo "Launching sandbox on $DISP_ID..."

Xephyr $DISP_ID -ac -screen 1355x1200 -title "Play Model" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

USER_DATA_DIR="$HOME/chrome-rl-play"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser in Sandbox..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 8

# --- Run the Play Script ---
echo ""
echo "======================================================="
echo "Sandbox ready. The agent will now start playing."
echo "Debug windows will appear on your MAIN screen."
echo "======================================================="
echo ""

# Run python on the CURRENT display, passing the sandbox ID and any other args
/home/netsky/Code/FitCats-rl/.venv/bin/python play_model.py --sandbox $DISP_ID "$@"

# --- Cleanup ---
echo "Session finished. Closing sandbox."
kill $XEPHYR_PID
