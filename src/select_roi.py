import cv2
import glob
import os

dataset_path = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

print("Total images:", len(image_files))

# Select SECOND image
file = image_files[1]

print("Using:", file)

img = cv2.imread(file)

if img is None:
    print("Could not read image")
    exit()

print("Image size:", img.shape[1], "x", img.shape[0])

# Resize only for displaying
display_width = 1200
scale = display_width / img.shape[1]
display_height = int(img.shape[0] * scale)

display_img = cv2.resize(img, (display_width, display_height))

# Select beam
roi = cv2.selectROI(
    "Select Beam",
    display_img,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select Beam")

x, y, w, h = roi

if w == 0 or h == 0:
    print("No ROI selected")
    exit()

# Convert coordinates to original image
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

print("\nROI Coordinates")
print("x =", x)
print("y =", y)
print("width =", w)
print("height =", h)

# Crop original high-resolution image
beam = img[y:y+h, x:x+w]

cv2.imshow("Beam ROI", beam)
cv2.waitKey(0)
cv2.destroyAllWindows()