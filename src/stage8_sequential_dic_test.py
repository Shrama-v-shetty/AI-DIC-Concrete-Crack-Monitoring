import cv2
import os
import re
import csv
import time
import numpy as np

from dic_engine import track_pair


# ============================================================
# PATHS
# ============================================================

BGSL_ROI_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

OUTPUT_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage8_Sequential_DIC_Test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEST RANGE
# ============================================================

START_FRAME = 1650
END_FRAME = 1700


# ============================================================
# EXTRACT FRAME NUMBER
# Example:
# BGSL-2-1650_0.tif  -> 1650
# ============================================================

def extract_frame_number(filename):

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    # Second-last number is the actual frame number
    return int(numbers[-2])


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = []

for filename in os.listdir(BGSL_ROI_DIR):

    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    frame_number = extract_frame_number(filename)

    if frame_number is None:
        continue

    if START_FRAME <= frame_number <= END_FRAME:
        image_files.append(
            (frame_number, os.path.join(BGSL_ROI_DIR, filename))
        )


# Sort by actual frame number
image_files.sort(key=lambda x: x[0])


print("\n============================================")
print("STAGE 8 — SEQUENTIAL DIC TEST")
print("============================================")

print("Requested range :", START_FRAME, "to", END_FRAME)
print("Images found    :", len(image_files))

if len(image_files) == 0:
    print("\nERROR: No images found.")
    raise SystemExit


print("\nAvailable frames:")
print([x[0] for x in image_files])


# ============================================================
# CHECK FOR MISSING FRAMES
# ============================================================

available_numbers = {x[0] for x in image_files}

missing_frames = []

for frame in range(START_FRAME, END_FRAME + 1):

    if frame not in available_numbers:
        missing_frames.append(frame)


if missing_frames:

    print("\nWARNING: Missing frames detected:")
    print(missing_frames)

else:

    print("\nNo missing frames in test range.")


# ============================================================
# PROCESS SEQUENTIAL PAIRS
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "sequential_summary.csv"
)


with open(summary_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Frame_1",
        "Frame_2",
        "Tracked_Points",
        "Average_ZNCC",
        "Minimum_ZNCC",
        "Average_dx",
        "Average_dy",
        "Average_Magnitude",
        "Maximum_Magnitude",
        "Processing_Time_seconds"
    ])


    total_pairs = len(image_files) - 1

    for pair_index in range(total_pairs):

        frame1, path1 = image_files[pair_index]
        frame2, path2 = image_files[pair_index + 1]


        # ----------------------------------------------------
        # Make sure they are actually consecutive
        # ----------------------------------------------------

        if frame2 != frame1 + 1:

            print(
                f"\nSkipping pair {frame1} -> {frame2}"
                f" because frames are not consecutive."
            )

            continue


        print("\n--------------------------------------------")
        print(
            f"Processing pair "
            f"{frame1} -> {frame2}"
        )
        print(
            f"Pair {pair_index + 1} / {total_pairs}"
        )
        print("--------------------------------------------")


        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        img1 = cv2.imread(
            path1,
            cv2.IMREAD_GRAYSCALE
        )

        img2 = cv2.imread(
            path2,
            cv2.IMREAD_GRAYSCALE
        )


        if img1 is None or img2 is None:

            print("ERROR: Could not read image.")

            continue


        # ----------------------------------------------------
        # Start timer
        # ----------------------------------------------------

        start_time = time.time()


        # ----------------------------------------------------
        # DIC
        # ----------------------------------------------------

        results = track_pair(
            img1,
            img2,
            subset_size=41,
            search_size=81,
            step=10
        )


        processing_time = time.time() - start_time


        # ----------------------------------------------------
        # No valid points
        # ----------------------------------------------------

        if len(results) == 0:

            print("No valid DIC points.")

            continue


        # ----------------------------------------------------
        # Convert results to NumPy
        #
        # columns:
        # X, Y, dx, dy, ZNCC
        # ----------------------------------------------------

        data = np.array(results)

        dx = data[:, 2]
        dy = data[:, 3]
        zncc = data[:, 4]

        magnitude = np.sqrt(
            dx ** 2 + dy ** 2
        )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        tracked_points = len(results)

        average_zncc = np.mean(zncc)
        minimum_zncc = np.min(zncc)

        average_dx = np.mean(dx)
        average_dy = np.mean(dy)

        average_magnitude = np.mean(magnitude)
        maximum_magnitude = np.max(magnitude)


        # ----------------------------------------------------
        # Save individual pair result
        # ----------------------------------------------------

        pair_filename = (
            f"dic_{frame1}_{frame2}.csv"
        )

        pair_path = os.path.join(
            OUTPUT_DIR,
            pair_filename
        )


        with open(pair_path, "w", newline="") as pair_file:

            pair_writer = csv.writer(pair_file)

            pair_writer.writerow([
                "X",
                "Y",
                "dx",
                "dy",
                "ZNCC",
                "Magnitude"
            ])


            for i in range(len(data)):

                pair_writer.writerow([
                    int(data[i, 0]),
                    int(data[i, 1]),
                    data[i, 2],
                    data[i, 3],
                    data[i, 4],
                    magnitude[i]
                ])


        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        writer.writerow([
            frame1,
            frame2,
            tracked_points,
            average_zncc,
            minimum_zncc,
            average_dx,
            average_dy,
            average_magnitude,
            maximum_magnitude,
            processing_time
        ])


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nResults:")
        print("Tracked points :", tracked_points)
        print("Average ZNCC   :", round(average_zncc, 6))
        print("Minimum ZNCC   :", round(minimum_zncc, 6))

        print("Average dx     :", round(average_dx, 6))
        print("Average dy     :", round(average_dy, 6))

        print(
            "Average magnitude :",
            round(average_magnitude, 6)
        )

        print(
            "Maximum magnitude :",
            round(maximum_magnitude, 6)
        )

        print(
            "Processing time   :",
            round(processing_time, 2),
            "seconds"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n============================================")
print("SEQUENTIAL DIC TEST COMPLETE")
print("============================================")

print("Output folder:")
print(OUTPUT_DIR)

print("\nSummary:")
print(summary_file)

print("\nDo NOT start the full 3900-image run yet.")
print("First check the sequential test results.")