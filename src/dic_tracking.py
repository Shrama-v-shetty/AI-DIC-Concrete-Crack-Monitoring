import cv2
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATH
# ============================================================

roi_path = r"C:\Users\shrama\Desktop\Dam updated dataset\dataset\BGSL_ROI"

output_csv = r"C:\Users\shrama\Desktop\Dam updated dataset\results\displacement.csv"

os.makedirs(
    os.path.dirname(output_csv),
    exist_ok=True
)

# ============================================================
# GET ROI FRAMES
# ============================================================

image_files = sorted(
    glob.glob(os.path.join(roi_path, "*.tif"))
)

print("==========================================")
print("          DIC-STYLE TRACKING")
print("==========================================")

print("Total ROI frames:", len(image_files))

if len(image_files) < 2:
    print("Not enough images for tracking.")
    exit()

# ============================================================
# REFERENCE FRAME
# ============================================================

reference = cv2.imread(
    image_files[0],
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    print("Could not read reference frame.")
    exit()

# ============================================================
# FEATURE DETECTION
# ============================================================

features = cv2.goodFeaturesToTrack(
    reference,
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)

if features is None:
    print("No trackable features found.")
    exit()

print("Features detected:", len(features))

# ============================================================
# TRACK FEATURES
# ============================================================

results = []

previous_frame = reference
previous_points = features

for i in range(1, len(image_files)):

    current_frame = cv2.imread(
        image_files[i],
        cv2.IMREAD_GRAYSCALE
    )

    if current_frame is None:
        print("Could not read:", image_files[i])
        continue

    # Optical flow
    current_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous_frame,
        current_frame,
        previous_points,
        None
    )

    if current_points is None:
        print("Tracking failed at frame:", i)
        continue

    # Keep successfully tracked points
    good_previous = previous_points[status == 1]
    good_current = current_points[status == 1]

    if len(good_current) == 0:
        print("No valid points at frame:", i)
        continue

    # Calculate displacement
    displacement = good_current - good_previous

    dx = np.mean(displacement[:, 0])
    dy = np.mean(displacement[:, 1])

    magnitude = np.sqrt(dx**2 + dy**2)

    filename = os.path.basename(image_files[i])

    results.append([
        i,
        filename,
        dx,
        dy,
        magnitude,
        len(good_current)
    ])

    # Prepare for next frame
    previous_frame = current_frame
    previous_points = good_current.reshape(-1, 1, 2)

    if i % 100 == 0:
        print(
            f"Processed {i}/{len(image_files)} "
            f"| Dx = {dx:.4f} "
            f"| Dy = {dy:.4f}"
        )

# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    results,
    columns=[
        "Frame",
        "Filename",
        "Displacement_X",
        "Displacement_Y",
        "Displacement_Magnitude",
        "Tracked_Points"
    ]
)

df.to_csv(
    output_csv,
    index=False
)

print("\n==========================================")
print("          TRACKING COMPLETE")
print("==========================================")

print("Results saved to:")
print(output_csv)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["Frame"],
    df["Displacement_Magnitude"]
)

plt.xlabel("Frame")
plt.ylabel("Displacement (pixels)")
plt.title("Beam Displacement vs Frame")

plt.grid(True)

plt.show()