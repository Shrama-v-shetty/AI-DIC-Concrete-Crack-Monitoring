import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re


# ==========================================
# PATHS
# ==========================================

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

DIC_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test"

OUTPUT_FOLDER = DIC_FOLDER

CSV_FILE = os.path.join(
    DIC_FOLDER,
    "dic_6599_6600.csv"
)


# ==========================================
# FIND FRAME 1273
# ==========================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


frame_file = None

for file in glob.glob(
    os.path.join(ROI_FOLDER, "*.tif")
):

    frame_number = get_frame_number(file)

    if frame_number == 6599:
        frame_file = file
        break


if frame_file is None:

    print("ERROR: Frame 1273 not found.")

    exit()


print("Using image:")
print(os.path.basename(frame_file))


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(
    frame_file,
    cv2.IMREAD_GRAYSCALE
)


if image is None:

    print("ERROR: Could not read image.")

    exit()


# ==========================================
# LOAD DIC DATA
# ==========================================

df = pd.read_csv(CSV_FILE)


print("\nDIC points:", len(df))

print(
    "Displacement range:",
    df["Displacement"].min(),
    "to",
    df["Displacement"].max()
)


# ==========================================
# LIMIT COLOR SCALE
# ==========================================

# Use 99th percentile so a few extreme points
# do not dominate the visualization.

vmax = np.percentile(
    df["Displacement"],
    99
)

vmin = 0


# ==========================================
# CREATE FIGURE
# ==========================================

plt.figure(
    figsize=(16, 7)
)


# ==========================================
# SHOW CONCRETE IMAGE
# ==========================================

plt.imshow(
    image,
    cmap="gray",
    origin="upper"
)


# ==========================================
# PLOT DIC DISPLACEMENT
# ==========================================

scatter = plt.scatter(
    df["X"],
    df["Y"],
    c=df["Displacement"],
    cmap="turbo",
    vmin=vmin,
    vmax=vmax,
    s=8,
    alpha=0.75
)


# ==========================================
# MAXIMUM DISPLACEMENT POINT
# ==========================================

max_index = df["Displacement"].idxmax()

max_point = df.loc[max_index]


plt.scatter(
    max_point["X"],
    max_point["Y"],
    facecolors="none",
    edgecolors="white",
    s=150,
    linewidths=2
)


plt.text(
    max_point["X"] + 15,
    max_point["Y"],
    f"Max: {max_point['Displacement']:.4f} px",
    color="white",
    fontsize=10,
    fontweight="bold"
)


# ==========================================
# COLORBAR
# ==========================================

plt.colorbar(
    scatter,
    label="DIC Displacement (pixels)"
)


# ==========================================
# LABELS
# ==========================================
plt.title(
    "DIC Displacement Field: Frame 6599 → 6600",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "X Position (pixels)"
)

plt.ylabel(
    "Y Position (pixels)"
)


plt.tight_layout()


# ==========================================
# SAVE
# ==========================================

output_file = os.path.join(
    OUTPUT_FOLDER,
    "dic_visualization_1273_1274.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


print("\n==========================================")
print("       DIC VISUALIZATION COMPLETE")
print("==========================================")

print("Saved:")
print(output_file)