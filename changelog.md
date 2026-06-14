# Changelog

This file documents the major changes and refactoring efforts applied to the `FitCats-rl` project.

## 2026-01-13: Hybrid Detection Model & Major Refactor

This series of changes represents a complete overhaul of the detection system, moving from a simple radius-based approach to a robust, hybrid "Detect, then Classify" model.

### Core Logic Overhaul
- **Switched to Dominant Color Classification**: Replaced the complex ORB feature matching system with a simpler, faster method based on **dominant color detection** (using k-means clustering). This approach leverages the distinct colors of cats in the game and is inherently rotation-invariant.
- **Improved Occlusion Resistance**: Modified the patch extraction logic to use a smaller, more central portion of the detected circle (**25%** of the radius). This focuses the classification on the core of the cat, making it more robust to partial occlusion by neighboring cats.
- **Switched to Hybrid Model**: The detection pipeline was completely refactored. The system now:
    1. Uses `HoughCircles` to find candidate circular objects on the main playfield.
    2. Extracts an image patch for each candidate.
    3. Crops the patch to its content to isolate the object from its background.
    4. Uses dominant color comparison to classify the object against a library of known dominant colors.

### Template Library Enhancements
- **Robust Uniqueness Checks**: The logic for discovering new templates was made more robust:
    - **Color-Insensitive**: New templates are considered unique if their dominant color is sufficiently different from all known template colors.
    - **Brightness-Insensitive**: Added **histogram equalization** before comparison to handle lighting variations and the "fade-in" effect of new cats. (Note: this is still used in `crop_to_content` implicitly for thresholding, but direct histogram comparison is removed from `is_new_color`).
    - **Blank-Insensitive**: Added a **standard deviation check** to automatically ignore blank or solid-color images, preventing them from being added to the template library.
- **Intuitive File Management**: The template library in the `cat_templates` directory is now managed intelligently:
    - **Sequential Re-indexing**: On startup, the script automatically renames template files `cat_*.png` to be perfectly sequential (0, 1, 2, ...).
    - **Safe Deletion**: Users can now safely delete or rename unwanted template files, and the script will correctly re-index the library on the next run.

### Autonomous Testing
- **Autonomous Clicks**: Added a feature to make the debug script click automatically in the center of the play area, allowing for hands-free testing of the detection and discovery pipeline.

### General Housekeeping
- **File Renaming**: Renamed `debug_cat_size.py` and `debug_cat_size.sh` to `debug_cat_count.py` and `debug_cat_count.sh` to better reflect their purpose.

## Previous Fixes

### Initial Debugging & Setup
- Fixed `DISPLAY` environment variable conflicts between `pyautogui`, `cv2`, and `mss` to ensure all tools targeted the correct screens (sandbox vs. main display).
- Corrected the `setup_agent.py` calibration script, which was not correctly targeting the sandbox display, preventing template capture from working.
- Fixed a `NameError` crash in `setup_agent.py`.

### Game Start Procedure (`utils.py`)
- Improved the responsiveness of the `start_game` function by reducing polling delay.
- Added a step to automatically click the "music" button on the main menu.
