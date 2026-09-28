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
# EXTRACT ACTUAL FRAME NUMBER FROM FILENAME
# Example:
# BGSL-2-0001_0.tif  -> 1
# BGSL-2-2526_0.tif  -> 2526
# BGSL-2-6600_0.tif  -> 6600
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    # The second-last number is the actual frame number
    return int(numbers[-2])


# ============================================================
# LOAD ROI IMAGES
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


# Sort by actual frame number
available_frames = sorted(frame_files.keys())


print("==========================================")
print("        STAGE 3 - DIC TEST")
print("==========================================")

print("ROI folder:")
print(ROI_FOLDER)

print("\nAvailable ROI frames:", len(available_frames))

if len(available_frames) > 0:
    print("First frame:", available_frames[0])
    print("Last frame :", available_frames[-1])


# ============================================================
# SELECT TEST PAIRS
# ============================================================
#
# IMPORTANT:
# These are consecutive ACTUAL experiment frames.
#
# We are NOT testing across the missing ranges.
#
# ============================================================

test_pairs = [
    (1, 2),
    (100, 101),
    (500, 501),
    (1000, 1001),
    (2000, 2001),
    (2525, 2526),
]


# ============================================================
# CHECK WHETHER TEST FRAMES EXIST
# ============================================================

valid_pairs = []

for frame1, frame2 in test_pairs:

    if frame1 in frame_files and frame2 in frame_files:

        valid_pairs.append((frame1, frame2))

    else:

        print(
            f"\nSkipping pair {frame1} -> {frame2} "
            f"(one or both images are unavailable)"
        )


print("\nTest pairs that will be processed:")

for frame1, frame2 in valid_pairs:

    print(f"  {frame1} -> {frame2}")


# ============================================================
# PROCESS EACH DIC PAIR
# ============================================================

summary = []


for frame1, frame2 in valid_pairs:

    print("\n==========================================")
    print(f"Processing Frame {frame1} -> Frame {frame2}")
    print("==========================================")

    file1 = frame_files[frame1]
    file2 = frame_files[frame2]

    print("Image 1:", os.path.basename(file1))
    print("Image 2:", os.path.basename(file2))

    # --------------------------------------------------------
    # Load grayscale images
    # --------------------------------------------------------

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
        continue

    if img2 is None:

        print("ERROR: Could not read:", file2)
        continue

    print("Image size:", img1.shape)

    # --------------------------------------------------------
    # Run existing DIC engine
    # --------------------------------------------------------

    results = track_pair(
        img1,
        img2
    )

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if len(results) == 0:

        print("WARNING: No DIC points were tracked.")

        continue

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Calculate displacement magnitude
    # --------------------------------------------------------

    df["Displacement"] = np.sqrt(
        df["dx"]**2 +
        df["dy"]**2
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_name = (
        f"dic_{frame1:04d}_{frame2:04d}.csv"
    )

    csv_path = os.path.join(
        OUTPUT_FOLDER,
        csv_name
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_dx = df["dx"].mean()
    average_dy = df["dy"].mean()

    average_displacement = (
        df["Displacement"].mean()
    )

    maximum_displacement = (
        df["Displacement"].max()
    )

    average_zncc = df["ZNCC"].mean()
    minimum_zncc = df["ZNCC"].min()

    maximum_zncc = df["ZNCC"].max()

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print("\nDIC RESULT")
    print("------------------------------------------")

    print(
        "Tracked points       :",
        len(df)
    )

    print(
        "Average dx           :",
        round(average_dx, 4)
    )

    print(
        "Average dy           :",
        round(average_dy, 4)
    )

    print(
        "Average displacement :",
        round(average_displacement, 4)
    )

    print(
        "Maximum displacement :",
        round(maximum_displacement, 4)
    )

    print(
        "Average ZNCC         :",
        round(average_zncc, 4)
    )

    print(
        "Minimum ZNCC         :",
        round(minimum_zncc, 4)
    )

    print(
        "Maximum ZNCC         :",
        round(maximum_zncc, 4)
    )

    print(
        "\nSaved:",
        csv_path
    )

    # --------------------------------------------------------
    # Add to summary
    # --------------------------------------------------------

    summary.append(
        [
            frame1,
            frame2,
            len(df),
            average_dx,
            average_dy,
            average_displacement,
            maximum_displacement,
            average_zncc,
            minimum_zncc,
            maximum_zncc
        ]
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

if len(summary) > 0:

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Frame1",
            "Frame2",
            "Tracked_Points",
            "Average_dx",
            "Average_dy",
            "Average_Displacement",
            "Maximum_Displacement",
            "Average_ZNCC",
            "Minimum_ZNCC",
            "Maximum_ZNCC"
        ]
    )

    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "stage3_dic_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n==========================================")
    print("          STAGE 3 TEST COMPLETE")
    print("==========================================")

    print(
        "\nSummary saved at:"
    )

    print(summary_path)

else:

    print("\n==========================================")
    print("       NO DIC RESULTS GENERATED")
    print("==========================================")