#!/bin/bash

# Define the Display ID for the sandbox (e.g., :99)
DISP_ID=":99"

echo "--- Starting Fit Cats Sandbox on $DISP_ID ---"

# 1. Start Xephyr (The nested window)
# -ac: Disable access control
# -screen: Set the resolution of the virtual monitor
Xephyr $DISP_ID -ac -screen 1280x960 &
XEPHYR_PID=$!
echo "Xephyr started (PID: $XEPHYR_PID)"

# Wait for it to initialize
sleep 2

# 2. Start Fluxbox (Window Manager)
# This allows the browser to be maximized inside the sandbox
DISPLAY=$DISP_ID fluxbox &
sleep 1

# 3. Start Chromium inside the sandbox
# We use a temporary user-data-dir so it doesn't conflict with your main browser
echo "Starting Browser..."
DISPLAY=$DISP_ID /snap/bin/chromium --user-data-dir=/tmp/chrome-rl-sandbox --no-first-run --start-maximized "https://sites.google.com/site/populardoodlegames/fit-cats" &

echo ""
echo "======================================================="
echo "Sandbox Ready!"
echo "1. Click inside the Xephyr window to focus it."
echo "2. You may need to re-run 'recapture_template.py' inside this sandbox."
echo ""
echo "To run the training script inside this sandbox, use:"
echo "sudo DISPLAY=$DISP_ID /home/netsky/Code/FitCats-rl/.venv/bin/python main.py"
echo "======================================================="
echo ""

# Keep script running to maintain the session
wait $XEPHYR_PID
