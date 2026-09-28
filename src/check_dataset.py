import os
import glob
import cv2

dataset_path = "../dataset/BGSL2"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

print("Total files:", len(image_files))

bad_files = []

for i, file in enumerate(image_files):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        bad_files.append((i, os.path.basename(file)))
    else:
        print(f"{i:03d}  {os.path.basename(file)}  {img.shape}")

print("\nUnreadable files:")
for item in bad_files:
    print(item)

print("\nTotal unreadable:", len(bad_files))