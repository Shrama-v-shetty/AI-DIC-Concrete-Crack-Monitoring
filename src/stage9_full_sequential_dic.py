import cv2
import os
import re
import csv
import time

from dic_engine_fast import track_pair_fast


# ============================================================
# PATHS
# ============================================================

ROI_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

OUTPUT_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage9_Full_Sequential_DIC"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DIC SETTINGS
# ============================================================

SUBSET_SIZE = 41
SEARCH_SIZE = 81
STEP = 10


# ============================================================
# EXTRACT ACTUAL FRAME NUMBER
#
# Example:
# BGSL-2-1651_0.tif -> 1651
#
# The second-last number is the actual image number.
# ============================================================

def get_frame_number(filename):

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# FIND ALL ROI IMAGES
# ============================================================

print("=" * 70)
print("FULL SEQUENTIAL DIC PROCESSING")
print("=" * 70)

print("\nScanning ROI images...")

image_files = []

for filename in os.listdir(ROI_DIR):

    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    frame_number = get_frame_number(filename)

    if frame_number is None:
        continue

    # Ignore frame 0
    if frame_number == 0:
        continue

    image_files.append((frame_number, filename))


# Sort by actual frame number
image_files.sort(key=lambda x: x[0])


print(f"Available images : {len(image_files)}")

if len(image_files) == 0:

    print("ERROR: No ROI images found.")

    raise SystemExit


# ============================================================
# CHECK FRAME RANGE
# ============================================================

first_frame = image_files[0][0]
last_frame = image_files[-1][0]

print(f"First frame      : {first_frame}")
print(f"Last frame       : {last_frame}")


# ============================================================
# CREATE FRAME LOOKUP
# ============================================================

frame_to_file = {
    frame: filename
    for frame, filename in image_files
}


# ============================================================
# CREATE LIST OF VALID CONSECUTIVE PAIRS
# ============================================================

pairs = []

for frame1 in sorted(frame_to_file.keys()):

    frame2 = frame1 + 1

    if frame2 in frame_to_file:

        pairs.append((frame1, frame2))


print(f"Consecutive pairs : {len(pairs)}")


# ============================================================
# REPORT MISSING RANGES
# ============================================================

available_frames = sorted(frame_to_file.keys())

missing_ranges = []

range_start = None
previous = None

for frame in available_frames:

    if previous is not None and frame > previous + 1:

        missing_ranges.append(
            (previous + 1, frame - 1)
        )

    previous = frame


if missing_ranges:

    print("\nMissing frame ranges:")

    for start, end in missing_ranges:

        print(f"  {start} - {end}")

else:

    print("\nNo missing frame ranges.")


# ============================================================
# OUTPUT SUMMARY FILE
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "full_dic_summary.csv"
)


summary_file = open(
    summary_path,
    "w",
    newline=""
)

writer = csv.writer(summary_file)

writer.writerow([
    "Frame_1",
    "Frame_2",
    "Tracked_Points",
    "Average_ZNCC",
    "Average_dx",
    "Average_dy",
    "Average_Magnitude",
    "Maximum_Magnitude",
    "Processing_Time_sec"
])


# ============================================================
# PROCESS PAIRS
# ============================================================

overall_start = time.time()

successful_pairs = 0
failed_pairs = 0


for pair_number, (frame1, frame2) in enumerate(pairs, start=1):

    print("\n" + "-" * 70)

    print(
        f"Pair {pair_number}/{len(pairs)} : "
        f"{frame1} -> {frame2}"
    )


    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    path1 = os.path.join(
        ROI_DIR,
        frame_to_file[frame1]
    )

    path2 = os.path.join(
        ROI_DIR,
        frame_to_file[frame2]
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

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

        failed_pairs += 1

        continue


    # --------------------------------------------------------
    # Run FAST DIC
    # --------------------------------------------------------

    start_time = time.time()

    try:

        results = track_pair_fast(
            img1,
            img2,
            subset_size=SUBSET_SIZE,
            search_size=SEARCH_SIZE,
            step=STEP
        )

    except Exception as e:

        print(f"ERROR during DIC: {e}")

        failed_pairs += 1

        continue


    processing_time = time.time() - start_time


    if len(results) == 0:

        print("WARNING: No valid tracked points.")

        failed_pairs += 1

        continue


    successful_pairs += 1


    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    zncc_values = [
        row[4]
        for row in results
    ]

    dx_values = [
        row[2]
        for row in results
    ]

    dy_values = [
        row[3]
        for row in results
    ]


    magnitudes = [

        (dx * dx + dy * dy) ** 0.5

        for dx, dy in zip(
            dx_values,
            dy_values
        )

    ]


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_zncc = (
        sum(zncc_values)
        / len(zncc_values)
    )

    average_dx = (
        sum(dx_values)
        / len(dx_values)
    )

    average_dy = (
        sum(dy_values)
        / len(dy_values)
    )

    average_magnitude = (
        sum(magnitudes)
        / len(magnitudes)
    )

    maximum_magnitude = max(
        magnitudes
    )


    # --------------------------------------------------------
    # Save individual DIC CSV
    # --------------------------------------------------------

    pair_csv = os.path.join(
        OUTPUT_DIR,
        f"dic_{frame1}_{frame2}.csv"
    )


    with open(
        pair_csv,
        "w",
        newline=""
    ) as f:

        pair_writer = csv.writer(f)

        pair_writer.writerow([
            "X",
            "Y",
            "dx",
            "dy",
            "ZNCC",
            "Magnitude"
        ])


        for row, magnitude in zip(
            results,
            magnitudes
        ):

            pair_writer.writerow([
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                magnitude
            ])


    # --------------------------------------------------------
    # Add summary row
    # --------------------------------------------------------

    writer.writerow([
        frame1,
        frame2,
        len(results),
        average_zncc,
        average_dx,
        average_dy,
        average_magnitude,
        maximum_magnitude,
        processing_time
    ])

    summary_file.flush()


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Tracked points       : {len(results)}"
    )

    print(
        f"Average ZNCC         : {average_zncc:.6f}"
    )

    print(
        f"Average dx           : {average_dx:.6f}"
    )

    print(
        f"Average dy           : {average_dy:.6f}"
    )

    print(
        f"Average magnitude    : {average_magnitude:.6f}"
    )

    print(
        f"Maximum magnitude    : {maximum_magnitude:.6f}"
    )

    print(
        f"Processing time      : {processing_time:.2f} sec"
    )


# ============================================================
# CLOSE SUMMARY
# ============================================================

summary_file.close()


# ============================================================
# FINAL INFORMATION
# ============================================================

total_time = time.time() - overall_start


print("\n")
print("=" * 70)
print("FULL SEQUENTIAL DIC COMPLETE")
print("=" * 70)

print(
    f"Successful pairs : {successful_pairs}"
)

print(
    f"Failed pairs     : {failed_pairs}"
)

print(
    f"Total time       : {total_time:.2f} seconds"
)

print(
    f"Total time       : {total_time / 60:.2f} minutes"
)

print("\nSummary:")
print(summary_path)

print("\nDIC results:")
print(OUTPUT_DIR)