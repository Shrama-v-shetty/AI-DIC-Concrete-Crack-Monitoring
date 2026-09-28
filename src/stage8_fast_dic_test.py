import cv2
import time
import numpy as np

from dic_engine_fast import track_pair_fast


# ============================================================
# TEST IMAGE PATHS
# ============================================================

IMG1_PATH = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\BGSL_ROI\BGSL-2-1651_0.tif"
)

IMG2_PATH = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\BGSL_ROI\BGSL-2-1652_0.tif"
)


# ============================================================
# LOAD IMAGES
# ============================================================

img1 = cv2.imread(
    IMG1_PATH,
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    IMG2_PATH,
    cv2.IMREAD_GRAYSCALE
)


if img1 is None:
    raise RuntimeError(
        "Could not read Frame 1651."
    )

if img2 is None:
    raise RuntimeError(
        "Could not read Frame 1652."
    )


print("\n============================================")
print("FAST DIC TEST")
print("============================================")

print("Frame 1 : 1651")
print("Frame 2 : 1652")
print("Image size :", img1.shape)


# ============================================================
# RUN DIC
# ============================================================

start_time = time.time()

results = track_pair_fast(
    img1,
    img2,
    subset_size=41,
    search_size=81,
    step=10
)

elapsed = time.time() - start_time


# ============================================================
# RESULTS
# ============================================================

if len(results) == 0:

    print("\nNo valid DIC points found.")
    raise SystemExit


data = np.array(results)

dx = data[:, 2]
dy = data[:, 3]
zncc = data[:, 4]

magnitude = np.sqrt(
    dx ** 2 + dy ** 2
)


print("\n========== FAST DIC RESULTS ==========")

print("Tracked points :", len(results))

print(
    "Average ZNCC   :",
    np.mean(zncc)
)

print(
    "Minimum ZNCC   :",
    np.min(zncc)
)

print(
    "Maximum ZNCC   :",
    np.max(zncc)
)

print(
    "\nAverage dx     :",
    np.mean(dx)
)

print(
    "Average dy     :",
    np.mean(dy)
)

print(
    "Average magnitude :",
    np.mean(magnitude)
)

print(
    "Maximum magnitude :",
    np.max(magnitude)
)

print(
    "\nProcessing time :",
    round(elapsed, 2),
    "seconds"
)

print("=======================================")


# ============================================================
# COMPARE WITH OLD RESULT
# ============================================================

print("\n========== OLD RESULT ==========")

print("Tracked points       : 8446")
print("Average ZNCC         : 0.997391")
print("Average magnitude    : 0.019732")
print("Maximum magnitude    : 0.086446")
print("Processing time      : 860.52 seconds")

print("================================")