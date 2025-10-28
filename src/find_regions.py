import mss
import cv2
import numpy as np
import time

print("Open Polytrack fullscreen and start playing!")
print("Taking screenshot in 5 seconds...")
time.sleep(5)

with mss.mss() as sct:
    # Capture full screen
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Draw grid lines to help locate regions
    height, width = img.shape[:2]
    
    # Draw lines every 100 pixels
    for i in range(0, width, 100):
        cv2.line(img, (i, 0), (i, height), (0, 255, 0), 1)
        cv2.putText(img, str(i), (i, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    for i in range(0, height, 100):
        cv2.line(img, (0, i), (width, i), (0, 255, 0), 1)
        cv2.putText(img, str(i), (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save
    cv2.imwrite('grid_screenshot.png', img)
    print(f"Saved grid_screenshot.png ({width}x{height})")
    print("Open it and tell me the EXACT pixel coordinates of:")
    print("1. Speed number (bottom right)")
    print("2. Time number (bottom center)")
