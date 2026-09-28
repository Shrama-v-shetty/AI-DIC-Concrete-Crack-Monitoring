import cv2
import glob
import os
import json
import pandas as pd

from dic_engine import track_pair

# ==========================================
# Load ROI
# ==========================================

with open("../output/beam_roi.json", "r") as f:
    roi = json.load(f)

roi_x = roi["x"]
roi_y = roi["y"]
roi_w = roi["width"]
roi_h = roi["height"]

print("Loaded ROI:")
print(roi)

# ==========================================
# Load Images
# ==========================================

dataset_path = "../dataset/BGSL2"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

valid_images = []

for file in image_files[2:]:

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        valid_images.append(img)

print("Total Images :", len(valid_images))
# ==========================================
# Process Image Pairs
# ==========================================

os.makedirs("../output", exist_ok=True)

print("\nReference Image : Image 3")
print("Starting Fixed Reference DIC...\n")

# Test only first 5 image pairs
for frame in range(1,40):

    print("\n====================================")
    print(f"Processing Pair {frame}/{len(valid_images)-1}")
    print(f"Image 3 -> Image {frame+3}")
    print("====================================")

    # Crop ROI from consecutive images
    # Reference image (Image 3) - always fixed
    img1 = valid_images[0][
    roi_y:roi_y+roi_h,
    roi_x:roi_x+roi_w
    ]

    # Current target image
    img2 = valid_images[frame][
    roi_y:roi_y+roi_h,
    roi_x:roi_x+roi_w
    ]

    # Run DIC
    results = track_pair(img1, img2)

    # Convert to DataFrame
    df = pd.DataFrame(
        results,
        columns=[
            "X",
            "Y",
            "dx",
            "dy",
            "ZNCC"
        ]
    )

    # Save one CSV for each image pair
    filename = f"../output/reference_0003_{frame+3:04d}.csv"

    df.to_csv(filename, index=False)

    print("Tracked Subsets :", len(df))
    print("Saved :", filename)

print("\n====================================")
print("Batch Processing Completed")
print("====================================")