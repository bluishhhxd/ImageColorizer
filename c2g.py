import cv2
import sys
import os

if len(sys.argv) < 2:
    print("Usage: py ColorToGrey.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ Error: File not found at '{image_path}'")
    sys.exit(1)

image=cv2.imread(image_path)
if image is None:
    print("❌ Error: Could not open image.")
    sys.exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(image_path, gray)

print(f"✅ Successfully converted '{image_path}' to grayscale (original overwritten).")
