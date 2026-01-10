from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import cv2
import numpy as np
import os
import shutil

def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--start-maximized')
    options.binary_location = "/snap/bin/chromium"

    driver = None
    system_driver_paths = ["/snap/bin/chromium.chromedriver", "/usr/bin/chromedriver", "/usr/local/bin/chromedriver"]
    for path in system_driver_paths:
        if os.path.exists(path) or shutil.which(path):
            try:
                service = Service(executable_path=path)
                driver = webdriver.Chrome(service=service, options=options)
                break
            except: pass

    if not driver:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://sites.google.com/site/populardoodlegames/fit-cats"
    driver.get(url)
    time.sleep(5)

    # 1. Navigate to Game Frame
    frame_path = []
    def find_play(current_path, depth=0):
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if "Play Fit Cats" in btn.text:
                    btn.click()
                    return True
        except: pass
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for i, iframe in enumerate(iframes):
            try:
                driver.switch_to.frame(iframe)
                current_path.append(i)
                if find_play(current_path, depth + 1): return True
                current_path.pop()
                driver.switch_to.parent_frame()
            except: driver.switch_to.parent_frame()
        return False

    if find_play(frame_path):
        print("Initial Play button clicked. Waiting 15s...")
        time.sleep(15)
        
        driver.switch_to.default_content()
        for i in frame_path:
            driver.switch_to.frame(driver.find_elements(By.TAG_NAME, "iframe")[i])
            
        # 2. Inspect the Game Frame
        print("Inspecting Game Frame...")
        
        # Save HTML
        with open("game_frame_source.html", "w") as f:
            f.write(driver.page_source)
        print("Saved game_frame_source.html")
        
        # Take screenshot
        driver.save_screenshot("debug_overlay.png")
        img = cv2.imread("debug_overlay.png")
        
        # Find Canvas
        canvas = None
        try:
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            for c in canvases:
                if c.size['width'] > 200:
                    canvas = c
                    rect = c.rect
                    print(f"Canvas found: {rect}")
                    # Draw Blue Box around Canvas
                    cv2.rectangle(img, 
                                  (int(rect['x']), int(rect['y'])), 
                                  (int(rect['x'] + rect['width']), int(rect['y'] + rect['height'])), 
                                  (255, 0, 0), 2)
                    break
        except: pass
        
        # Find Potential Overlays (Divs/Images with high Z-index or absolute position)
        print("Searching for overlays...")
        elements = driver.find_elements(By.CSS_SELECTOR, "div, img, button, a")
        for el in elements:
            try:
                if not el.is_displayed(): continue
                
                rect = el.rect
                if rect['width'] < 20 or rect['height'] < 20: continue
                
                # Check if it overlaps with canvas (if canvas found)
                is_overlay = False
                if canvas:
                    c_rect = canvas.rect
                    # Simple overlap check
                    if (rect['x'] < c_rect['x'] + c_rect['width'] and
                        rect['x'] + rect['width'] > c_rect['x'] and
                        rect['y'] < c_rect['y'] + c_rect['height'] and
                        rect['y'] + rect['height'] > c_rect['y']):
                        is_overlay = True
                
                if is_overlay:
                    print(f"Overlay found: {el.tag_name} id='{el.get_attribute('id')}' class='{el.get_attribute('class')}'")
                    # Draw Red Box around Overlays
                    cv2.rectangle(img, 
                                  (int(rect['x']), int(rect['y'])), 
                                  (int(rect['x'] + rect['width']), int(rect['y'] + rect['height'])), 
                                  (0, 0, 255), 2)
            except: pass
            
        cv2.imwrite("debug_overlay_annotated.png", img)
        print("Saved debug_overlay_annotated.png. Please check this image.")

    driver.quit()

if __name__ == "__main__":
    main()
