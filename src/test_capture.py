import mss
import cv2
import numpy as np
import time

print("Switch to Polytrack window NOW!")
print("Capturing in 5 seconds...")
time.sleep(5)

# Capture full screen
with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Save it
    cv2.imwrite('screen_test.png', img)
    print(f"Screenshot saved! Resolution: {img.shape[1]}x{img.shape[0]}")
    print("Check screen_test.png")
