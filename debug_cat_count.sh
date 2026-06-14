#!/bin/bash

echo "--- Cat Count Debug Sandbox ---"

# --- Launch Sandbox ---
DISP_ID=":99"
echo "Launching sandbox on $DISP_ID..."

Xephyr $DISP_ID -ac -screen 1355x1200 -title "Cat Count Debugger" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

USER_DATA_DIR="$HOME/chrome-rl-cat-debug"
mkdir -p "$USER_DATA_DIR"


# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "Error: jq is not installed. Please install it using 'sudo apt-get install jq' or your distribution's package manager."
    exit 1
fi

# Read game_url from config.json
GAME_URL=$(jq -r '.game_url' config.json)

echo "Starting Browser in Sandbox..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "$GAME_URL" &
sleep 8

# --- Run the Debug Script ---
echo ""
echo "======================================================="
echo "Sandbox ready. The cat counting debugger will now start."
echo "The debug window will appear on your MAIN screen."
echo "Play the game in the sandbox window."
echo "======================================================="
echo ""

# Run python on the CURRENT display (usually :0), passing the sandbox ID
/home/netsky/Code/FitCats-rl/.venv/bin/python debug_cat_count.py --sandbox $DISP_ID

# --- Cleanup ---
echo "Debug session finished. Closing sandbox."
kill $XEPHYR_PID
