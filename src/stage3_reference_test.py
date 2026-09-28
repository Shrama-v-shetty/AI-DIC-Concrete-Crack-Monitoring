import cv2
import os
import re
import glob
import pandas as pd
import numpy as np

from dic_engine import track_pair


# ==========================================
# PATHS
# ==========================================

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# REFERENCE AND FINAL FRAME
# ==========================================

REFERENCE_FRAME = 1
FINAL_FRAME = 6600


# ==========================================
# FIND FRAME NUMBER
# ==========================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ==========================================
# FIND IMAGES
# ==========================================

print("\n==========================================")
print("      FRAME 1 -> FRAME 6600 DIC TEST")
print("==========================================")

frame_files = {}

all_files = glob.glob(
    os.path.join(ROI_FOLDER, "*.tif")
)

for file in all_files:

    frame_number = get_frame_number(file)

    if frame_number is None:
        continue

    if frame_number == 0:
        continue

    frame_files[frame_number] = file


print("\nAvailable ROI frames:", len(frame_files))


if REFERENCE_FRAME not in frame_files:

    print("ERROR: Frame 1 not found.")
    exit()


if FINAL_FRAME not in frame_files:

    print("ERROR: Frame 6600 not found.")
    exit()


file1 = frame_files[REFERENCE_FRAME]
file2 = frame_files[FINAL_FRAME]


print("\nReference:")
print(os.path.basename(file1))

print("\nFinal:")
print(os.path.basename(file2))


# ==========================================
# LOAD IMAGES
# ==========================================

img1 = cv2.imread(
    file1,
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    file2,
    cv2.IMREAD_GRAYSCALE
)


if img1 is None or img2 is None:

    print("ERROR: Could not read one of the images.")
    exit()


print("\nImage size:", img1.shape)


# ==========================================
# PREPARE IMAGES FOR REGISTRATION
# ==========================================

print("\n==========================================")
print("      COARSE IMAGE REGISTRATION")
print("==========================================")

# Convert to float
template = img1.astype(np.float32) / 255.0
moving = img2.astype(np.float32) / 255.0


# ==========================================
# ECC AFFINE REGISTRATION
# ==========================================

warp_matrix = np.eye(
    2,
    3,
    dtype=np.float32
)

criteria = (
    cv2.TERM_CRITERIA_EPS |
    cv2.TERM_CRITERIA_COUNT,
    100,
    1e-6
)


try:

    correlation, warp_matrix = cv2.findTransformECC(
        template,
        moving,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        5
    )

    print("\nECC registration successful.")

    print(
        "ECC correlation:",
        correlation
    )

except cv2.error as e:

    print("\nECC registration failed.")

    print(e)

    print(
        "\nWe will stop here rather than produce",
        "a misleading DIC result."
    )

    exit()


# ==========================================
# ALIGN FINAL IMAGE
# ==========================================

aligned_img2 = cv2.warpAffine(
    img2,
    warp_matrix,
    (
        img1.shape[1],
        img1.shape[0]
    ),
    flags=cv2.INTER_LINEAR |
          cv2.WARP_INVERSE_MAP
)


# ==========================================
# SAVE ALIGNMENT CHECK
# ==========================================

alignment_path = os.path.join(
    OUTPUT_FOLDER,
    "alignment_1_6600.png"
)


comparison = np.hstack(
    (
        img1,
        aligned_img2
    )
)


cv2.imwrite(
    alignment_path,
    comparison
)


print("\nAlignment comparison saved:")
print(alignment_path)


# ==========================================
# RUN EXISTING DIC ENGINE
# ==========================================

print("\n==========================================")
print("          RUNNING DIC")
print("==========================================")

print(
    "\nFrame 1 -> Frame 6600"
)

print(
    "DIC is being calculated after coarse alignment..."
)

results = track_pair(
    img1,
    aligned_img2
)


# ==========================================
# CHECK RESULTS
# ==========================================

if len(results) == 0:

    print("\n==========================================")
    print("          NO DIC POINTS FOUND")
    print("==========================================")

    exit()


# ==========================================
# DATAFRAME
# ==========================================

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


# ==========================================
# RESIDUAL DISPLACEMENT
# ==========================================

df["Residual_Displacement"] = np.sqrt(
    df["dx"] ** 2 +
    df["dy"] ** 2
)


# ==========================================
# SAVE CSV
# ==========================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "dic_1_6600_registered.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ==========================================
# STATISTICS
# ==========================================

print("\n==========================================")
print("          FINAL DIC RESULT")
print("==========================================")

print(
    "Tracked points:",
    len(df)
)

print(
    "Average residual dx:",
    round(df["dx"].mean(), 6)
)

print(
    "Average residual dy:",
    round(df["dy"].mean(), 6)
)

print(
    "Average residual displacement:",
    round(
        df["Residual_Displacement"].mean(),
        6
    )
)

print(
    "Maximum residual displacement:",
    round(
        df["Residual_Displacement"].max(),
        6
    )
)

print(
    "Average ZNCC:",
    round(
        df["ZNCC"].mean(),
        6
    )
)

print(
    "Minimum ZNCC:",
    round(
        df["ZNCC"].min(),
        6
    )
)

print(
    "Maximum ZNCC:",
    round(
        df["ZNCC"].max(),
        6
    )
)


# ==========================================
# MAXIMUM RESIDUAL POINT
# ==========================================

max_index = df[
    "Residual_Displacement"
].idxmax()

max_point = df.loc[max_index]


print("\n==========================================")
print("       MAXIMUM RESIDUAL POINT")
print("==========================================")

print(
    "X:",
    max_point["X"]
)

print(
    "Y:",
    max_point["Y"]
)

print(
    "dx:",
    max_point["dx"]
)

print(
    "dy:",
    max_point["dy"]
)

print(
    "Residual displacement:",
    max_point["Residual_Displacement"]
)

print(
    "ZNCC:",
    max_point["ZNCC"]
)


print("\n==========================================")
print("             TEST COMPLETE")
print("==========================================")

print("\nAlignment:")
print(alignment_path)

print("\nDIC CSV:")
print(csv_path)