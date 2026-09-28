import cv2
import glob
import os
import numpy as np

# -----------------------
# Read first image
# -----------------------

dataset_path = "../dataset/BGSL2"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# -----------------------
# Parameters
# -----------------------

subset_size = 31
step = 20

variance_threshold = 800

# -----------------------
# Generate subsets
# -----------------------

count = 0

for y in range(0, img.shape[0] - subset_size, step):

    for x in range(0, img.shape[1] - subset_size, step):

        subset = img[y:y+subset_size, x:x+subset_size]

        variance = np.var(subset)

        if variance > variance_threshold:

            cv2.rectangle(
                output,
                (x, y),
                (x+subset_size, y+subset_size),
                (0,255,0),
                1
            )

            count += 1

print("Good subsets:", count)

cv2.imshow("Selected DIC Subsets", output)

cv2.waitKey(0)
cv2.destroyAllWindows()