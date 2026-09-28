import cv2
import glob
import json
import numpy as np
import pandas as pd
import os

# ==========================================
# Base Directories
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "BGSL2")

# ==========================================
# Overlay Function
# ==========================================

def generate_overlay(strain_csv, output_image):

    # ==========================================
    # Load ROI
    # ==========================================

    roi_path = os.path.join(OUTPUT_DIR, "beam_roi.json")

    with open(roi_path, "r") as f:
        roi = json.load(f)

    roi_x = roi["x"]
    roi_y = roi["y"]
    roi_w = roi["width"]
    roi_h = roi["height"]

    # ==========================================
    # Load Reference Beam Image
    # ==========================================

    image_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.tif")))

    valid_images = []

    for file in image_files[2:]:

        img = cv2.imread(file)

        if img is not None:
            valid_images.append(img)

    beam = valid_images[0]

    beam = beam[
        roi_y:roi_y + roi_h,
        roi_x:roi_x + roi_w
    ]

    print("Beam Size :", beam.shape)

    # ==========================================
    # Load Strain CSV
    # ==========================================

    df = pd.read_csv(strain_csv)

    x_values = sorted(df["X"].unique())
    y_values = sorted(df["Y"].unique())

    strain_grid = np.zeros((len(y_values), len(x_values)))

    for _, row in df.iterrows():

        x = x_values.index(row["X"])
        y = y_values.index(row["Y"])

        # Absolute strain
        strain_grid[y, x] = abs(row["eps_yy"])

    print("Grid Shape :", strain_grid.shape)

    # ==========================================
    # Resize
    # ==========================================

    heatmap = cv2.resize(
        strain_grid,
        (beam.shape[1], beam.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    # ==========================================
    # Smooth Heatmap
    # ==========================================

    heatmap = cv2.GaussianBlur(
        heatmap,
        (7, 7),
        0
    )

    # ==========================================
    # Maximum Strain
    # ==========================================

    max_strain = np.max(heatmap)

    max_index = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape
    )

    max_y, max_x = max_index

    print("\nMaximum Strain :", max_strain)
    print("Location :", max_x, max_y)

    # ==========================================
    # Severity
    # ==========================================

    if max_strain < 0.002:
        severity = "LOW"

    elif max_strain < 0.01:
        severity = "MEDIUM"

    else:
        severity = "HIGH"

    print("Severity :", severity)

    # ==========================================
    # Normalize
    # ==========================================

    heatmap_norm = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap_norm = heatmap_norm.astype(np.uint8)

    # ==========================================
    # Color Heatmap
    # ==========================================

    color_heatmap = cv2.applyColorMap(
        heatmap_norm,
        cv2.COLORMAP_JET
    )

    # ==========================================
    # Overlay
    # ==========================================

    overlay = cv2.addWeighted(
        beam,
        0.65,
        color_heatmap,
        0.35,
        0
    )

    # ==========================================
    # Draw Maximum Strain Point
    # ==========================================

    cv2.circle(
        overlay,
        (max_x, max_y),
        12,
        (0, 0, 255),
        3
    )

    cv2.putText(
        overlay,
        "Max Strain",
        (max_x + 20, max_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        overlay,
        f"Severity : {severity}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ==========================================
    # Save
    # ==========================================

    cv2.imwrite(output_image, overlay)

    print("\nOverlay Saved Successfully")
    print("Saved At :", os.path.abspath(output_image))
    print("File Exists :", os.path.exists(output_image))


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":

    strain_csv = os.path.join(
        OUTPUT_DIR,
        "strain.csv"
    )

    output_image = os.path.join(
        OUTPUT_DIR,
        "overlay.png"
    )

    generate_overlay(
        strain_csv,
        output_image
    )