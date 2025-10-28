import mss
import cv2
import numpy as np
import time
import pytesseract
import re

CONFIG = {
    "speed_region": {"top": 900, "left": 1300, "width": 110, "height": 56},
    "time_region": {"top": 900, "left": 600, "width": 270, "height": 56},
}

print("Switch to Polytrack and start a race!")
print("Reading speed/time in 5 seconds...")
time.sleep(5)

with mss.mss() as sct:
    for i in range(10):
        # Capture speed
        speed_img = sct.grab(CONFIG["speed_region"])
        speed_frame = np.array(speed_img)
        speed_gray = cv2.cvtColor(speed_frame, cv2.COLOR_BGRA2GRAY)
        _, speed_thresh = cv2.threshold(speed_gray, 150, 255, cv2.THRESH_BINARY)
        
        # Capture time
        time_img = sct.grab(CONFIG["time_region"])
        time_frame = np.array(time_img)
        time_gray = cv2.cvtColor(time_frame, cv2.COLOR_BGRA2GRAY)
        _, time_thresh = cv2.threshold(time_gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        speed_text = pytesseract.image_to_string(speed_thresh, config='--psm 7 digits')
        time_text = pytesseract.image_to_string(time_thresh, config='--psm 7')
        
        print(f"Speed: {speed_text.strip()} | Time: {time_text.strip()}")
        
        # Save images for debugging
        if i == 0:
            cv2.imwrite('speed_region.png', speed_thresh)
            cv2.imwrite('time_region.png', time_thresh)
            print("Saved speed_region.png and time_region.png")
        
        time.sleep(0.5)