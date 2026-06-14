#!/bin/bash

DISP_ID=":99"

echo "--- Launching OCR Debug Sandbox on $DISP_ID ---"

# Start Xephyr and redirect its output to /dev/null
Xephyr $DISP_ID -ac -screen 1355x1200 -title "OCR Debug Sandbox" > /dev/null 2>&1 &
XEPHYR_PID=$!
sleep 2

if command -v fluxbox &> /dev/null
then
    DISPLAY=$DISP_ID fluxbox &
    sleep 1
fi

USER_DATA_DIR="$HOME/chrome-rl-ocr-debug"
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
echo "Sandbox Ready for OCR Debugging!"
echo "Running debug_ocr.py inside the sandbox..."
echo "You can start playing the game in the sandbox window."
echo "Then, press Enter in the terminal where you ran this script."
echo "======================================================="
echo ""

# Run the OCR debug tool inside the sandbox
DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python debug_ocr.py

# Cleanup
echo "Debug session finished. Closing sandbox."
kill $XEPHYR_PID
