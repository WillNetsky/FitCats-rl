import cv2
import sys

def main():
    # Load the image
    img_path = "game_title.png"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load {img_path}. Make sure analyze_game_ui.py ran at least once.")
        return

    print("Instructions:")
    print("1. A window will open showing the game screenshot.")
    print("2. CLICK on the 'Play' button (or any other UI element you want to measure).")
    print("3. The coordinates will be printed in this terminal.")
    print("4. Press any key to close the window.")

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: X={x}, Y={y}")
            # Draw a marker
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Find Coordinates', img)

    cv2.imshow('Find Coordinates', img)
    cv2.setMouseCallback('Find Coordinates', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
