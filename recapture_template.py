import pyautogui
import mss
import numpy as np
import cv2
import os
import time

def select_roi(img, window_name="Select ROI"):
    print(f"\nIn the '{window_name}' window, drag a box around the target area.")
    print("Press ENTER to confirm your selection, or 'c' to cancel.")
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi

def capture_template(prompt, output_filename, window_title):
    print(f"\n--- {window_title} ---")
    print(prompt)
    choice = input(f"Press ENTER to capture, or 'n' to skip if you have a good '{output_filename}': ").lower()
    
    if choice == 'n':
        print(f"Skipping '{output_filename}'.")
        return None

    with mss.mss() as sct:
        # In the sandbox, monitor[1] is the virtual screen
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    roi = select_roi(screenshot_bgr, window_title)
    if not roi:
        print("Selection cancelled.")
        return "abort"
        
    x, y, w, h = roi
    template_img = screenshot_bgr[y:y+h, x:x+w]
    cv2.imwrite(output_filename, template_img)
    print(f"Saved '{output_filename}'.")
    return template_img

def main():
    print("=== Comprehensive Template Recapture Tool ===")
    
    # --- 0. Newgrounds Start Button ---
    result = capture_template(
        "If you see a 'Click to Play' or 'Launch Game' button overlay, capture it now.",
        "template_newgrounds_play.png",
        "Step 0: Select the NEWGROUNDS PLAY BUTTON"
    )
    if isinstance(result, str) and result == "abort": return

    if result is not None:
        print("Clicking the button to proceed...")
        # We can't easily click here without coordinates, so we ask the user
        input("Please manually click the button in the browser to load the game, then press Enter...")
        time.sleep(5) # Wait for load

    # --- 1. Title Screen ---
    title_img_result = capture_template(
        "Make sure the game's TITLE SCREEN is clearly visible.",
        "game_title.png",
        "Step 1: Select the ENTIRE GAME WINDOW"
    )
    
    if isinstance(title_img_result, str) and title_img_result == "abort": return
    elif title_img_result is None:
        if not os.path.exists("game_title.png"):
            print("Skipped, but 'game_title.png' does not exist. Aborting.")
            return
        title_img = cv2.imread("game_title.png")
    else:
        title_img = title_img_result

    # --- 2. Play Button ---
    print("\n--- Step 1b: Select Play Button ---")
    print("Now, select the PLAY button from within the title screen image.")
    play_roi = select_roi(title_img.copy(), "Select the PLAY button")
    if play_roi:
        x, y, w, h = play_roi
        template_play_img = title_img[y:y+h, x:x+w]
        cv2.imwrite("template_play.png", template_play_img)
        print("Saved 'template_play.png'.")
    else:
        print("Skipping Play button.")

    # --- 3. Empty Board ---
    result = capture_template(
        "Manually start a game so the board is empty.",
        "template_empty_board.png",
        "Step 2: Select a STATIC portion of the ACTIVE game (e.g., the empty board)"
    )
    if isinstance(result, str) and result == "abort": return

    # --- 4. Restart Button ---
    result = capture_template(
        "Play the game until you get a GAME OVER screen.",
        "template_restart.png",
        "Step 3: Select the RESTART button"
    )
    if isinstance(result, str) and result == "abort": return

    print("\nAll templates have been processed successfully!")

if __name__ == "__main__":
    main()
