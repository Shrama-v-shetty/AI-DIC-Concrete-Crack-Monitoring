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

CSV_FILE = os.path.join(
    DIC_FOLDER,
    "dic_1273_1274.csv"
)

OUTPUT_FILE = os.path.join(
    DIC_FOLDER,
    "dic_corrected_1273_1274.png"
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

    if frame_number == 1273:
        frame_file = file
        break


if frame_file is None:

    print("ERROR: Frame 1273 not found.")
    exit()


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
# LOAD DIC RESULTS
# ==========================================

df = pd.read_csv(CSV_FILE)


print("\n==========================================")
print("       CORRECTED DIC VISUALIZATION")
print("==========================================")

print("DIC points:", len(df))


# ==========================================
# REMOVE GLOBAL TRANSLATION
# ==========================================

# The median is used because it is robust
# against a few unusually high displacement
# points.

global_dx = df["dx"].median()
global_dy = df["dy"].median()

print("\nEstimated global movement:")
print("Global dx:", global_dx)
print("Global dy:", global_dy)


df["dx_corrected"] = (
    df["dx"] - global_dx
)

df["dy_corrected"] = (
    df["dy"] - global_dy
)


# ==========================================
# CORRECTED DISPLACEMENT
# ==========================================

df["Corrected_Displacement"] = np.sqrt(
    df["dx_corrected"] ** 2 +
    df["dy_corrected"] ** 2
)


print(
    "\nCorrected average displacement:",
    df["Corrected_Displacement"].mean()
)

print(
    "Corrected maximum displacement:",
    df["Corrected_Displacement"].max()
)


# ==========================================
# REMOVE BOTTOM FIXTURE REGION
# ==========================================

# Your ROI image contains a dark experimental
# fixture near the bottom-center.
#
# We do NOT modify the original image.
# We only exclude its DIC points here.

fixture_mask = (
    (df["X"] >= 900) &
    (df["X"] <= 1050) &
    (df["Y"] >= 500)
)


print(
    "\nDIC points removed from fixture region:",
    fixture_mask.sum()
)


df_valid = df[~fixture_mask].copy()


print(
    "Remaining valid DIC points:",
    len(df_valid)
)


# ==========================================
# FIND MAXIMUM VALID DISPLACEMENT
# ==========================================

max_index = df_valid[
    "Corrected_Displacement"
].idxmax()

max_point = df_valid.loc[max_index]


print("\n==========================================")
print("      MAXIMUM VALID DEFORMATION")
print("==========================================")

print("X:", max_point["X"])
print("Y:", max_point["Y"])

print(
    "Corrected displacement:",
    max_point["Corrected_Displacement"]
)

print(
    "ZNCC:",
    max_point["ZNCC"]
)


# ==========================================
# COLOR SCALE
# ==========================================

vmax = np.percentile(
    df_valid["Corrected_Displacement"],
    99
)


# ==========================================
# CREATE FIGURE
# ==========================================

plt.figure(
    figsize=(16, 7)
)


# ==========================================
# SHOW ORIGINAL CONCRETE
# ==========================================

plt.imshow(
    image,
    cmap="gray",
    origin="upper"
)


# ==========================================
# PLOT CORRECTED DIC FIELD
# ==========================================

scatter = plt.scatter(
    df_valid["X"],
    df_valid["Y"],
    c=df_valid["Corrected_Displacement"],
    cmap="turbo",
    vmin=0,
    vmax=vmax,
    s=8,
    alpha=0.75
)


# ==========================================
# MARK MAXIMUM VALID POINT
# ==========================================

plt.scatter(
    max_point["X"],
    max_point["Y"],
    facecolors="none",
    edgecolors="white",
    s=180,
    linewidths=2
)


plt.text(
    max_point["X"] + 15,
    max_point["Y"],
    f"Max: {max_point['Corrected_Displacement']:.4f} px",
    color="white",
    fontsize=10,
    fontweight="bold"
)


# ==========================================
# COLORBAR
# ==========================================

plt.colorbar(
    scatter,
    label="Corrected DIC Displacement (pixels)"
)


# ==========================================
# TITLE
# ==========================================

plt.title(
    "Corrected DIC Displacement Field: 1273 → 1274",
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

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


print("\n==========================================")
print("       VISUALIZATION COMPLETE")
print("==========================================")

print("Saved:")
print(OUTPUT_FILE)