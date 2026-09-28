import cv2
import os
import re
import glob
import pandas as pd
import numpy as np

from dic_engine import track_pair


# ============================================================
# PATHS
# ============================================================

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# GET ACTUAL FRAME NUMBER FROM FILENAME
# Example:
# BGSL-2-1274_0.tif -> 1274
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# FIND ALL ROI IMAGES
# ============================================================

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


available_frames = sorted(frame_files.keys())


# ============================================================
# CHECK REQUIRED FRAMES
# ============================================================

FRAME1 = 1000
FRAME2 = 1274

print("==========================================")
print("      STAGE 3 - DIC 1000 -> 1274")
print("==========================================")

print("ROI folder:")
print(ROI_FOLDER)

print("\nTotal available ROI frames:", len(available_frames))

if FRAME1 not in frame_files:
    print("\nERROR: Frame 1000 is not available.")
    raise SystemExit

if FRAME2 not in frame_files:
    print("\nERROR: Frame 1274 is not available.")
    raise SystemExit


# ============================================================
# LOAD IMAGES
# ============================================================

file1 = frame_files[FRAME1]
file2 = frame_files[FRAME2]

print("\nFrame 1:", os.path.basename(file1))
print("Frame 2:", os.path.basename(file2))


img1 = cv2.imread(
    file1,
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    file2,
    cv2.IMREAD_GRAYSCALE
)


if img1 is None:
    print("ERROR: Could not read:", file1)
    raise SystemExit

if img2 is None:
    print("ERROR: Could not read:", file2)
    raise SystemExit


print("\nImage size:", img1.shape)


# ============================================================
# RUN DIC
# ============================================================

print("\nRunning DIC...")
print("This may take some time.")


results = track_pair(
    img1,
    img2
)


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    print("\nWARNING: No DIC points were tracked.")
    raise SystemExit


# ============================================================
# CREATE DATAFRAME
# ============================================================

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


# Calculate displacement magnitude

df["Displacement"] = np.sqrt(
    df["dx"]**2 +
    df["dy"]**2
)


# ============================================================
# STATISTICS
# ============================================================

average_dx = df["dx"].mean()
average_dy = df["dy"].mean()

average_displacement = df["Displacement"].mean()
maximum_displacement = df["Displacement"].max()
minimum_displacement = df["Displacement"].min()

average_zncc = df["ZNCC"].mean()
minimum_zncc = df["ZNCC"].min()
maximum_zncc = df["ZNCC"].max()


# ============================================================
# MAXIMUM DISPLACEMENT POINT
# ============================================================

max_index = df["Displacement"].idxmax()

max_point = df.loc[max_index]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("             DIC RESULT")
print("==========================================")

print("Tracked points       :", len(df))

print("Average dx           :", round(average_dx, 6))
print("Average dy           :", round(average_dy, 6))

print("Average displacement :", round(
    average_displacement, 6
))

print("Maximum displacement :", round(
    maximum_displacement, 6
))

print("Minimum displacement :", round(
    minimum_displacement, 6
))

print("\nAverage ZNCC         :", round(
    average_zncc, 6
))

print("Minimum ZNCC         :", round(
    minimum_zncc, 6
))

print("Maximum ZNCC         :", round(
    maximum_zncc, 6
))


print("\n==========================================")
print("      MAXIMUM DISPLACEMENT POINT")
print("==========================================")

print("X :", max_point["X"])
print("Y :", max_point["Y"])

print("dx :", max_point["dx"])
print("dy :", max_point["dy"])

print(
    "Displacement :",
    max_point["Displacement"]
)

print(
    "ZNCC :",
    max_point["ZNCC"]
)


# ============================================================
# SAVE CSV
# ============================================================

csv_name = "dic_1000_1274.csv"

csv_path = os.path.join(
    OUTPUT_FOLDER,
    csv_name
)

df.to_csv(
    csv_path,
    index=False
)


print("\n==========================================")
print("              FILE SAVED")
print("==========================================")

print(csv_path)

print("\nStage 3 test completed.")