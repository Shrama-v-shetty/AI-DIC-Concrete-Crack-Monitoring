import cv2
import glob
import os
import numpy as np

# ------------------------------
# Load first valid image
# ------------------------------

dataset_path = "../dataset/BGSL2"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

img = None

for file in image_files[2:]:      # Start from 3rd image
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        break

if img is None:
    print("No image found")
    exit()

# ------------------------------
# Parameters
# ------------------------------

subset_size = 31
step = 20

half = subset_size // 2

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

good_subsets = 0

# ------------------------------
# Scan the image
# ------------------------------

for y in range(half, img.shape[0]-half, step):

    for x in range(half, img.shape[1]-half, step):

        subset = img[
            y-half:y+half+1,
            x-half:x+half+1
        ]

        variance = np.var(subset)

        # Good subset
        if variance > 500:

            good_subsets += 1

            cv2.rectangle(
                output,
                (x-half, y-half),
                (x+half, y+half),
                (0,255,0),
                1
            )

print("Good subsets:", good_subsets)

cv2.imshow("Subset Quality", output)

cv2.waitKey(0)
cv2.destroyAllWindows()