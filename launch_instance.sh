#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./launch_instance.sh <INSTANCE_ID>"
    exit 1
fi

ID=$1
DISP_ID=":9$ID"

echo "--- Launching Instance $ID on $DISP_ID ---"

# 1. Start Xephyr
# Increased width to 1355 and height to 1200
Xephyr $DISP_ID -ac -screen 1355x1200 -title "FitCats Agent $ID" &
XEPHYR_PID=$!
sleep 2

# 2. Start Fluxbox
DISPLAY=$DISP_ID fluxbox &
sleep 1

# 3. Start Browser
USER_DATA_DIR="$HOME/chrome-rl-instance-$ID"
mkdir -p "$USER_DATA_DIR"

echo "Starting Browser..."
DISPLAY=$DISP_ID /snap/bin/chromium \
    --user-data-dir="$USER_DATA_DIR" \
    --no-first-run \
    --start-maximized \
    --no-sandbox \
    "https://www.newgrounds.com/portal/view/913713" &
sleep 8

# 4. Start Training
echo "Starting AI..."
DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python main.py --id $ID

# Cleanup
kill $XEPHYR_PID
