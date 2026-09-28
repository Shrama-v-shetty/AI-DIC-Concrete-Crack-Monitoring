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

OUTPUT_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage8_Sequential_DIC_Test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEST RANGE
# ============================================================

START_FRAME = 1651
END_FRAME = 1701


# ============================================================
# GET FRAME NUMBER FROM FILENAME
# Example:
# BGSL-2-1651_0.tif -> 1651
# ============================================================

def get_frame_number(filename):

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = []

for filename in os.listdir(ROI_DIR):

    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    frame_number = get_frame_number(filename)

    if frame_number is None:
        continue

    if START_FRAME <= frame_number <= END_FRAME:
        image_files.append((frame_number, filename))


# Sort according to actual frame number
image_files.sort(key=lambda x: x[0])


print("=" * 60)
print("FAST SEQUENTIAL DIC TEST")
print("=" * 60)

print(f"Start frame : {START_FRAME}")
print(f"End frame   : {END_FRAME}")
print(f"Images found: {len(image_files)}")


# ============================================================
# CHECK FOR MISSING FRAMES
# ============================================================

found_numbers = [x[0] for x in image_files]

expected_numbers = list(range(START_FRAME, END_FRAME + 1))

missing = [x for x in expected_numbers if x not in found_numbers]

if missing:

    print("\nWARNING: Missing frames:")
    print(missing)

else:

    print("All frames are available.")


# ============================================================
# DICTIONARY FOR QUICK IMAGE LOOKUP
# ============================================================

frame_to_file = {
    frame: filename
    for frame, filename in image_files
}


# ============================================================
# SUMMARY CSV
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "sequential_dic_summary.csv"
)


summary_rows = []


# ============================================================
# PROCESS EACH CONSECUTIVE PAIR
# ============================================================

total_pairs = 0
successful_pairs = 0

overall_start = time.time()


for frame1 in range(START_FRAME, END_FRAME):

    frame2 = frame1 + 1

    print("\n" + "-" * 60)
    print(f"Processing pair {frame1} -> {frame2}")

    # --------------------------------------------------------
    # Check that both images exist
    # --------------------------------------------------------

    if frame1 not in frame_to_file or frame2 not in frame_to_file:

        print("SKIPPED: One or both images are missing.")

        continue


    file1 = os.path.join(
        ROI_DIR,
        frame_to_file[frame1]
    )

    file2 = os.path.join(
        ROI_DIR,
        frame_to_file[frame2]
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)


    if img1 is None or img2 is None:

        print("SKIPPED: Could not read image.")

        continue


    # --------------------------------------------------------
    # Run FAST DIC
    # --------------------------------------------------------

    pair_start = time.time()

    results = track_pair_fast(
        img1,
        img2,
        subset_size=41,
        search_size=81,
        step=10
    )

    pair_time = time.time() - pair_start


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_pairs += 1

    if len(results) == 0:

        print("WARNING: No tracked points.")

        continue


    successful_pairs += 1


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
        for dx, dy in zip(dx_values, dy_values)
    ]


    avg_zncc = sum(zncc_values) / len(zncc_values)

    avg_dx = sum(dx_values) / len(dx_values)

    avg_dy = sum(dy_values) / len(dy_values)

    avg_magnitude = sum(magnitudes) / len(magnitudes)

    max_magnitude = max(magnitudes)


    # --------------------------------------------------------
    # Save individual pair CSV
    # --------------------------------------------------------

    pair_csv = os.path.join(
        OUTPUT_DIR,
        f"dic_{frame1}_{frame2}.csv"
    )


    with open(pair_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "X",
            "Y",
            "dx",
            "dy",
            "ZNCC",
            "Magnitude"
        ])


        for row, magnitude in zip(results, magnitudes):

            writer.writerow([
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                magnitude
            ])


    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary_rows.append([
        frame1,
        frame2,
        len(results),
        avg_zncc,
        avg_dx,
        avg_dy,
        avg_magnitude,
        max_magnitude,
        pair_time
    ])


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(f"Tracked points : {len(results)}")
    print(f"Average ZNCC   : {avg_zncc:.6f}")
    print(f"Average dx     : {avg_dx:.6f}")
    print(f"Average dy     : {avg_dy:.6f}")
    print(f"Average magnitude : {avg_magnitude:.6f}")
    print(f"Maximum magnitude : {max_magnitude:.6f}")
    print(f"Processing time : {pair_time:.2f} seconds")


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(summary_file, "w", newline="") as f:

    writer = csv.writer(f)

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

    writer.writerows(summary_rows)


# ============================================================
# FINAL STATISTICS
# ============================================================

overall_time = time.time() - overall_start


print("\n")
print("=" * 60)
print("SEQUENTIAL TEST COMPLETE")
print("=" * 60)

print(f"Pairs processed successfully : {successful_pairs}")
print(f"Total pair attempts          : {total_pairs}")

if successful_pairs > 0:

    average_pair_time = (
        sum(row[8] for row in summary_rows)
        / len(summary_rows)
    )

    print(f"Average time per pair        : {average_pair_time:.2f} seconds")

print(f"Total processing time        : {overall_time:.2f} seconds")

print("\nSummary saved to:")
print(summary_file)

print("\nOutput folder:")
print(OUTPUT_DIR)