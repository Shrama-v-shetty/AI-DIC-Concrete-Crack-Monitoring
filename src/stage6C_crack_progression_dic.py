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

ROI_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\BGSL_ROI"
)

OUTPUT_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6C_Crack_Progression"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FUNCTION TO GET ACTUAL FRAME NUMBER
# ============================================================

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


# ============================================================
# FIND AVAILABLE IMAGES
# ============================================================

all_files = glob.glob(
    os.path.join(ROI_FOLDER, "*.tif")
)

frame_files = {}

for file in all_files:

    frame = get_frame_number(file)

    if frame is None:
        continue

    # Ignore frame 0
    if frame == 0:
        continue

    frame_files[frame] = file


available_frames = sorted(frame_files.keys())


print("==========================================")
print("   STAGE 6C - CRACK PROGRESSION DIC")
print("==========================================")

print(
    "\nAvailable ROI frames:",
    len(available_frames)
)


# ============================================================
# DIC TEST INTERVALS
# ============================================================

test_pairs = [

    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432)

]


# ============================================================
# CHECK PAIRS
# ============================================================

valid_pairs = []

print("\nChecking DIC intervals...")

for frame1, frame2 in test_pairs:

    if frame1 in frame_files and frame2 in frame_files:

        valid_pairs.append(
            (frame1, frame2)
        )

        print(
            f"  OK: {frame1} -> {frame2}"
        )

    else:

        print(
            f"  SKIP: {frame1} -> {frame2}"
        )


# ============================================================
# SUMMARY STORAGE
# ============================================================

summary = []


# ============================================================
# PROCESS EACH INTERVAL
# ============================================================

for frame1, frame2 in valid_pairs:

    print("\n==========================================")
    print(
        f"Processing DIC: "
        f"Frame {frame1} -> Frame {frame2}"
    )
    print("==========================================")


    file1 = frame_files[frame1]
    file2 = frame_files[frame2]


    print(
        "Image 1:",
        os.path.basename(file1)
    )

    print(
        "Image 2:",
        os.path.basename(file2)
    )


    # --------------------------------------------------------
    # READ IMAGES
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

        print(
            "ERROR: Could not read:",
            file1
        )

        continue


    if img2 is None:

        print(
            "ERROR: Could not read:",
            file2
        )

        continue


    print(
        "Image size:",
        img1.shape
    )


    # --------------------------------------------------------
    # RUN DIC
    # --------------------------------------------------------

    results = track_pair(
        img1,
        img2
    )


    if len(results) == 0:

        print(
            "WARNING: No DIC points tracked."
        )

        continue


    # --------------------------------------------------------
    # CREATE DATAFRAME
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


    # Calculate displacement magnitude

    df["Displacement"] = np.sqrt(
        df["dx"] ** 2 +
        df["dy"] ** 2
    )


    # --------------------------------------------------------
    # SAVE DIC POINT DATA
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
    # CALCULATE STATISTICS
    # --------------------------------------------------------

    tracked_points = len(df)

    average_dx = df["dx"].mean()
    average_dy = df["dy"].mean()

    average_displacement = (
        df["Displacement"].mean()
    )

    maximum_displacement = (
        df["Displacement"].max()
    )

    percentile_95 = (
        df["Displacement"].quantile(0.95)
    )

    percentile_99 = (
        df["Displacement"].quantile(0.99)
    )

    average_zncc = (
        df["ZNCC"].mean()
    )

    minimum_zncc = (
        df["ZNCC"].min()
    )

    maximum_zncc = (
        df["ZNCC"].max()
    )


    # --------------------------------------------------------
    # LOCATION OF MAXIMUM DISPLACEMENT
    # --------------------------------------------------------

    max_row = df.loc[
        df["Displacement"].idxmax()
    ]

    max_x = max_row["X"]
    max_y = max_row["Y"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nDIC RESULT")
    print("------------------------------------------")

    print(
        "Tracked points       :",
        tracked_points
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
        "95th percentile      :",
        round(percentile_95, 4)
    )

    print(
        "99th percentile      :",
        round(percentile_99, 4)
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
        "Maximum displacement location:"
    )

    print(
        "    X =",
        round(max_x, 2)
    )

    print(
        "    Y =",
        round(max_y, 2)
    )

    print(
        "\nSaved:",
        csv_path
    )


    # --------------------------------------------------------
    # ADD TO SUMMARY
    # --------------------------------------------------------

    summary.append([
        frame1,
        frame2,
        tracked_points,
        average_dx,
        average_dy,
        average_displacement,
        maximum_displacement,
        percentile_95,
        percentile_99,
        average_zncc,
        minimum_zncc,
        maximum_zncc,
        max_x,
        max_y
    ])


# ============================================================
# SAVE SUMMARY
# ============================================================

if len(summary) > 0:

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Frame_Start",
            "Frame_End",
            "Tracked_Points",
            "Average_dx",
            "Average_dy",
            "Average_Displacement",
            "Maximum_Displacement",
            "Displacement_95th",
            "Displacement_99th",
            "Average_ZNCC",
            "Minimum_ZNCC",
            "Maximum_ZNCC",
            "Max_Displacement_X",
            "Max_Displacement_Y"
        ]
    )


    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "crack_progression_dic_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print("\n==========================================")
    print("      STAGE 6C TEST COMPLETE")
    print("==========================================")


    print(
        "\nSummary saved:"
    )

    print(
        summary_path
    )


    print("\nSUMMARY")
    print("------------------------------------------")

    print(
        summary_df.to_string(
            index=False
        )
    )


else:

    print("\n==========================================")
    print("       NO DIC RESULTS GENERATED")
    print("==========================================")