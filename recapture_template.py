import pyautogui
import mss
import numpy as np
import cv2

def select_roi(img, window_name="Select ROI"):
    """Lets the user select a Region of Interest in an image."""
    print(f"\nIn the '{window_name}' window, drag a box around the target area.")
    print("Press ENTER to confirm your selection.")
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi

def main():
    print("=== Comprehensive Template Recapture Tool ===")
    print("This will capture all necessary templates in one go.")

    # --- 1. Capture Game Title / Main Window ---
    print("\nStep 1: Capture the entire game window.")
    print("Make sure the game's TITLE SCREEN is clearly visible on your primary monitor.")
    input("Press Enter to take a screenshot of your primary monitor...")

    with mss.mss() as sct:
        primary_monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(primary_monitor))
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    game_roi = select_roi(screenshot_bgr, "Select the ENTIRE GAME WINDOW")
    if not game_roi:
        print("No game window selected. Aborting.")
        return
    
    x_game, y_game, w_game, h_game = game_roi
    game_title_img = screenshot_bgr[y_game:y_game+h_game, x_game:x_game+w_game]
    cv2.imwrite("game_title.png", game_title_img)
    print("Saved 'game_title.png'.")

    # --- 2. Capture Play Button from the Title Image ---
    print("\nStep 2: Capture the PLAY button from the image you just selected.")
    play_roi = select_roi(game_title_img.copy(), "Select the PLAY button")
    if not play_roi:
        print("No play button selected. Aborting.")
        return
        
    x, y, w, h = play_roi
    template_play_img = game_title_img[y:y+h, x:x+w]
    cv2.imwrite("template_play.png", template_play_img)
    print("Saved 'template_play.png'.")

    # --- 3. Capture Empty Board ---
    print("\nStep 3: Capture the EMPTY PLAYING FIELD.")
    print("Manually click the 'Play' button in the game window now.")
    input("Once the game has started with an empty board, press Enter...")

    with mss.mss() as sct:
        # Re-grab the whole monitor to ensure we see the updated game state
        primary_monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(primary_monitor))
        empty_board_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    empty_board_roi = select_roi(empty_board_bgr.copy(), "Select the EMPTY PLAYING FIELD")
    if not empty_board_roi:
        print("No empty board selected. Aborting.")
        return
        
    x, y, w, h = empty_board_roi
    template_empty_board_img = empty_board_bgr[y:y+h, x:x+w]
    cv2.imwrite("template_empty_board.png", template_empty_board_img)
    print("Saved 'template_empty_board.png'.")

    # --- 4. Capture Restart Button ---
    print("\nStep 4: Capture the RESTART button.")
    print("Play the game manually until you reach the GAME OVER screen.")
    input("Press Enter when the GAME OVER screen is visible...")

    with mss.mss() as sct:
        # Re-grab the whole monitor again
        primary_monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(primary_monitor))
        game_over_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    restart_roi = select_roi(game_over_bgr.copy(), "Select the RESTART button")
    if not restart_roi:
        print("No restart button selected. Aborting.")
        return

    x, y, w, h = restart_roi
    template_restart_img = game_over_bgr[y:y+h, x:x+w]
    cv2.imwrite("template_restart.png", template_restart_img)
    print("Saved 'template_restart.png'.")

    print("\nAll templates have been recaptured successfully!")
    print("You can now run 'python main.py'.")

if __name__ == "__main__":
    main()
