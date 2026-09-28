import cv2
import glob
import os
import json
import pandas as pd

# =====================================
# Load ROI
# =====================================

with open("../output/beam_roi.json", "r") as f:
    roi = json.load(f)

roi_x = roi["x"]
roi_y = roi["y"]
roi_w = roi["width"]
roi_h = roi["height"]

# =====================================
# Load Reference Image (Image 3)
# =====================================

dataset_path = "../dataset/BGSL2"

image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))

img = None

for file in image_files[2:]:

    img = cv2.imread(file)

    if img is not None:
        break

# Crop beam ROI
beam = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# =====================================
# Load displacement CSV
# =====================================

df = pd.read_csv("../output/displacement.csv")

# =====================================
# Draw displacement vectors
# =====================================

output = beam.copy()

scale = 10      # Arrow scale

for _, row in df.iterrows():

    x = int(row["X"])
    y = int(row["Y"])

    dx = float(row["dx"])
    dy = float(row["dy"])

    start = (x, y)
    end = (
        int(x + dx * scale),
        int(y + dy * scale)
    )

    cv2.arrowedLine(
        output,
        start,
        end,
        (0, 0, 255),
        1,
        tipLength=0.3
    )

# =====================================
# Save Result
# =====================================

os.makedirs("../output", exist_ok=True)

cv2.imwrite(
    "../output/displacement_vectors.png",
    output
)

print("Saved: output/displacement_vectors.png")

# =====================================
# Display
# =====================================

cv2.imshow("Displacement Field", output)

cv2.waitKey(0)
cv2.destroyAllWindows()