import cv2
import glob
import os

# Dataset path
dataset_path = "../dataset/BGSL2"

# Read first valid image
image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

img = None

for file in image_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        break

if img is None:
    print("No valid image found!")
    exit()

subset_size = 31
step = 20

subsets = []

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

half = subset_size // 2

for y in range(half, img.shape[0] - half, step):
    for x in range(half, img.shape[1] - half, step):

        subsets.append((x, y))

        cv2.rectangle(
            output,
            (x-half, y-half),
            (x+half, y+half),
            (0,255,0),
            1
        )

print("Total subsets:", len(subsets))

cv2.imshow("Subset Grid", output)

cv2.waitKey(0)
cv2.destroyAllWindows()