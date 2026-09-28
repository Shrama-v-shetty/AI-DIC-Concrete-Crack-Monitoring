import cv2
import os
import re
import glob
import pandas as pd
import numpy as np

from dic_engine import track_pair


# ==========================================
# FOLDERS
# ==========================================

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# FRAMES TO TEST
# ==========================================

FRAME_1 = 6500
FRAME_2 = 6600


# ==========================================
# GET ACTUAL FRAME NUMBER
# ==========================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    # Example:
    # BGSL-2-1274_0.tif
    #
    # numbers = ['2', '1274', '0']
    #
    # Actual frame = second-last number

    return int(numbers[-2])


# ==========================================
# FIND ROI IMAGES
# ==========================================

print("\n==========================================")
print("       CRACK INITIATION DIC TEST")
print("==========================================")

print("\nSearching ROI images...")

all_files = glob.glob(
    os.path.join(ROI_FOLDER, "*.tif")
)

frame_files = {}

for file in all_files:

    frame_number = get_frame_number(file)

    if frame_number is None:
        print("Could not identify frame:", file)
        continue

    # Ignore frame 0
    if frame_number == 0:
        continue

    frame_files[frame_number] = file


print("Total available ROI frames:", len(frame_files))


# ==========================================
# CHECK REQUIRED FRAMES
# ==========================================

if FRAME_1 not in frame_files:

    print("\nERROR:")
    print("Frame", FRAME_1, "was not found.")

    exit()


if FRAME_2 not in frame_files:

    print("\nERROR:")
    print("Frame", FRAME_2, "was not found.")

    exit()


file1 = frame_files[FRAME_1]
file2 = frame_files[FRAME_2]


print("\nFrame 1:")
print(os.path.basename(file1))

print("\nFrame 2:")
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


if img1 is None:

    print("\nERROR: Could not read:")
    print(file1)

    exit()


if img2 is None:

    print("\nERROR: Could not read:")
    print(file2)

    exit()


print("\nImage size:", img1.shape)


# ==========================================
# RUN DIC
# ==========================================

print("\n==========================================")
print("Starting DIC tracking")
print("==========================================")

print("\nFrame:", FRAME_1, "->", FRAME_2)
print("Please wait...\n")


results = track_pair(
    img1,
    img2
)


# ==========================================
# CHECK RESULTS
# ==========================================

if len(results) == 0:

    print("\n==========================================")
    print("       NO DIC POINTS FOUND")
    print("==========================================")

    exit()


# ==========================================
# CREATE DATAFRAME
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
# CALCULATE DISPLACEMENT
# ==========================================

df["Displacement"] = np.sqrt(
    df["dx"] ** 2 +
    df["dy"] ** 2
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==========================================")
print("             DIC RESULT")
print("==========================================")

print(
    "Tracked points       :",
    len(df)
)

print(
    "Average dx           :",
    round(df["dx"].mean(), 6)
)

print(
    "Average dy           :",
    round(df["dy"].mean(), 6)
)

print(
    "Average displacement :",
    round(df["Displacement"].mean(), 6)
)

print(
    "Maximum displacement :",
    round(df["Displacement"].max(), 6)
)

print(
    "Minimum displacement :",
    round(df["Displacement"].min(), 6)
)

print(
    "Average ZNCC         :",
    round(df["ZNCC"].mean(), 6)
)

print(
    "Minimum ZNCC         :",
    round(df["ZNCC"].min(), 6)
)

print(
    "Maximum ZNCC         :",
    round(df["ZNCC"].max(), 6)
)


# ==========================================
# FIND MAXIMUM DISPLACEMENT LOCATION
# ==========================================

max_index = df["Displacement"].idxmax()

max_point = df.loc[max_index]

print("\n==========================================")
print("      MAXIMUM DISPLACEMENT POINT")
print("==========================================")

print(
    "X                  :",
    max_point["X"]
)

print(
    "Y                  :",
    max_point["Y"]
)

print(
    "dx                 :",
    max_point["dx"]
)

print(
    "dy                 :",
    max_point["dy"]
)

print(
    "Displacement       :",
    max_point["Displacement"]
)

print(
    "ZNCC               :",
    max_point["ZNCC"]
)


# ==========================================
# SAVE DIC CSV
# ==========================================

csv_name = (
    f"dic_{FRAME_1:04d}_{FRAME_2:04d}.csv"
)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    csv_name
)

df.to_csv(
    csv_path,
    index=False
)


print("\n==========================================")
print("             FILE SAVED")
print("==========================================")

print(csv_path)

print("\n==========================================")
print("       CRACK DIC TEST COMPLETE")
print("==========================================")