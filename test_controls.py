from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os
import shutil

def main():
    print("Initializing Chrome Options...")
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--start-maximized')
    options.binary_location = "/snap/bin/chromium"

    print("Attempting to find compatible ChromeDriver...")
    driver = None
    system_driver_paths = ["/snap/bin/chromium.chromedriver", "/usr/bin/chromedriver", "/usr/local/bin/chromedriver"]
    
    for path in system_driver_paths:
        if os.path.exists(path) or shutil.which(path):
            print(f"Found system ChromeDriver at: {path}")
            try:
                service = Service(executable_path=path)
                driver = webdriver.Chrome(service=service, options=options)
                print("Successfully started with system ChromeDriver.")
                break
            except Exception as e:
                print(f"Failed to start with {path}: {e}")

    if not driver:
        print("Could not start WebDriver.")
        return

    url = "https://sites.google.com/site/populardoodlegames/fit-cats"
    print(f"Navigating to {url}...")
    driver.get(url)
    
    time.sleep(5)

    # 1. Find and Click Play Button
    print("\n--- Searching for Play Button ---")
    frame_path = []
    
    def find_and_click_play(current_path, depth=0):
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if "Play Fit Cats" in btn.text:
                    print(f"Found 'Play Fit Cats' button at depth {depth}. Clicking...")
                    btn.click()
                    return True
        except:
            pass

        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for i, iframe in enumerate(iframes):
            try:
                driver.switch_to.frame(iframe)
                current_path.append(i)
                if find_and_click_play(current_path, depth + 1):
                    return True
                current_path.pop()
                driver.switch_to.parent_frame()
            except:
                driver.switch_to.parent_frame()
        return False

    if find_and_click_play(frame_path):
        print(f"Play button clicked in frame path: {frame_path}")
        print("Waiting 15s for game to load...")
        time.sleep(15)

        print("Navigating back to game frame...")
        driver.switch_to.default_content()
        for i in frame_path:
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            driver.switch_to.frame(iframes[i])

        print("Searching for canvas (including Shadow DOM)...")
        
        def expand_shadow_element(element):
            shadow_root = driver.execute_script('return arguments[0].shadowRoot', element)
            return shadow_root

        def find_canvas_advanced(depth=0):
            # 1. Standard search
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            for c in canvases:
                w = c.size['width']
                h = c.size['height']
                print(f"Found canvas {w}x{h} at depth {depth}")
                if w > 100: return c

            # 2. Shadow DOM search
            # Get all elements and check for shadow root
            # This is expensive, so we limit it to likely containers
            all_elements = driver.find_elements(By.CSS_SELECTOR, "div, section, app-root, game-container")
            for el in all_elements:
                try:
                    shadow = expand_shadow_element(el)
                    if shadow:
                        print(f"Found Shadow Root at depth {depth}")
                        # Search inside shadow
                        # Note: Selenium 4 can search in shadow root directly
                        try:
                            shadow_canvases = shadow.find_elements(By.TAG_NAME, "canvas")
                            for c in shadow_canvases:
                                w = c.size['width']
                                h = c.size['height']
                                print(f"Found Shadow Canvas {w}x{h}")
                                if w > 100: return c
                        except:
                            pass
                except:
                    pass

            # 3. Recurse into iframes
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                driver.switch_to.frame(iframe)
                c = find_canvas_advanced(depth + 1)
                if c: return c
                driver.switch_to.parent_frame()
            return None

        game_canvas = find_canvas_advanced()
        
        if game_canvas:
            print("Canvas found!")
            w = game_canvas.size['width']
            h = game_canvas.size['height']
            print(f"Canvas size: {w}x{h}")

            actions = ActionChains(driver)
            print("Clicking center to start game...")
            actions.move_to_element_with_offset(game_canvas, w/2, h/2).click().perform()
            time.sleep(2)
            
            for _ in range(5):
                x = random.randint(0, w)
                y = random.randint(0, h)
                print(f"Clicking at {x}, {y}")
                actions.move_to_element_with_offset(game_canvas, x, y).click().perform()
                time.sleep(0.5)
        else:
            print("Canvas NOT found.")
            driver.save_screenshot("debug_screenshot.png")
            with open("debug_frame.html", "w") as f:
                f.write(driver.page_source)

    else:
        print("Warning: Play button not found.")

    print("Test complete. Press Enter to close...")
    input()
    driver.quit()

if __name__ == "__main__":
    main()
