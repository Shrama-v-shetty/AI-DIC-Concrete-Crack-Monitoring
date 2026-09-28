import os
import pandas as pd
import numpy as np


# ============================================================
# STAGE 19C - DIC ENGINE VALIDATION
# Compare Stage 9 validated DIC against Stage 19B DIC
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

STAGE9_DIR = os.path.join(
    BASE_DIR,
    "Stage9_Full_Sequential_DIC"
)

STAGE19B_DIR = os.path.join(
    BASE_DIR,
    "Stage19B_DIC_Results",
    "frames_1274_1276"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage19C_DIC_Validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TEST PAIRS
# ============================================================

TEST_PAIRS = [
    (1274, 1275),
    (1275, 1276)
]


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STAGE 19C - DIC ENGINE VALIDATION")
print("=" * 75)

print()
print("IMPORTANT:")
print(
    "Stage 9 was validated using the ROI image dataset."
)

print(
    "Stage 19B must therefore be tested against the "
    "same ROI images for a fair comparison."
)


# ============================================================
# LOAD ONE DIC CSV
# ============================================================

def load_dic(path):

    if not os.path.exists(path):

        return None

    df = pd.read_csv(path)

    return df


# ============================================================
# CALCULATE FIELD STATISTICS
# ============================================================

def calculate_statistics(df):

    if df is None or df.empty:

        return None

    required = [
        "X",
        "Y",
        "dx",
        "dy"
    ]

    for column in required:

        if column not in df.columns:

            return None

    dx = pd.to_numeric(
        df["dx"],
        errors="coerce"
    )

    dy = pd.to_numeric(
        df["dy"],
        errors="coerce"
    )

    if "ZNCC" in df.columns:

        zncc = pd.to_numeric(
            df["ZNCC"],
            errors="coerce"
        )

    else:

        zncc = pd.Series(
            np.nan,
            index=df.index
        )


    valid = (
        dx.notna()
        & dy.notna()
    )

    dx = dx[valid].to_numpy(
        dtype=float
    )

    dy = dy[valid].to_numpy(
        dtype=float
    )

    zncc = zncc[valid].to_numpy(
        dtype=float
    )


    magnitude = np.sqrt(
        dx ** 2
        + dy ** 2
    )


    return {
        "Tracked_Points": len(dx),

        "Average_ZNCC": (
            float(np.nanmean(zncc))
            if np.isfinite(zncc).any()
            else np.nan
        ),

        "Average_dx": float(
            np.mean(dx)
        ),

        "Average_dy": float(
            np.mean(dy)
        ),

        "Average_Magnitude": float(
            np.mean(magnitude)
        ),

        "Maximum_Magnitude": float(
            np.max(magnitude)
        )
    }


# ============================================================
# COMPARE
# ============================================================

results = []


for frame1, frame2 in TEST_PAIRS:

    print()
    print("=" * 65)

    print(
        f"PAIR: {frame1} -> {frame2}"
    )


    stage9_path = os.path.join(
        STAGE9_DIR,
        f"dic_{frame1}_{frame2}.csv"
    )

    stage19b_path = os.path.join(
        STAGE19B_DIR,
        f"dic_{frame1}_{frame2}.csv"
    )


    print()
    print(
        "Stage 9:"
    )

    print(
        stage9_path
    )

    print()
    print(
        "Stage 19B:"
    )

    print(
        stage19b_path
    )


    stage9 = load_dic(
        stage9_path
    )

    stage19b = load_dic(
        stage19b_path
    )


    stats9 = calculate_statistics(
        stage9
    )

    stats19b = calculate_statistics(
        stage19b
    )


    if stats9 is None:

        print(
            "Could not read Stage 9 result."
        )

        continue


    if stats19b is None:

        print(
            "Could not read Stage 19B result."
        )

        continue


    print()
    print(
        "STAGE 9 VALIDATED"
    )

    print(
        f"Tracked points : "
        f"{stats9['Tracked_Points']}"
    )

    print(
        f"Average ZNCC   : "
        f"{stats9['Average_ZNCC']:.6f}"
    )

    print(
        f"Average dx     : "
        f"{stats9['Average_dx']:.6f}"
    )

    print(
        f"Average dy     : "
        f"{stats9['Average_dy']:.6f}"
    )

    print(
        f"Average mag    : "
        f"{stats9['Average_Magnitude']:.6f}"
    )

    print(
        f"Maximum mag    : "
        f"{stats9['Maximum_Magnitude']:.6f}"
    )


    print()
    print(
        "STAGE 19B"
    )

    print(
        f"Tracked points : "
        f"{stats19b['Tracked_Points']}"
    )

    print(
        f"Average ZNCC   : "
        f"{stats19b['Average_ZNCC']:.6f}"
    )

    print(
        f"Average dx     : "
        f"{stats19b['Average_dx']:.6f}"
    )

    print(
        f"Average dy     : "
        f"{stats19b['Average_dy']:.6f}"
    )

    print(
        f"Average mag    : "
        f"{stats19b['Average_Magnitude']:.6f}"
    )

    print(
        f"Maximum mag    : "
        f"{stats19b['Maximum_Magnitude']:.6f}"
    )


    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------

    point_difference = (
        stats19b["Tracked_Points"]
        - stats9["Tracked_Points"]
    )

    zncc_difference = (
        stats19b["Average_ZNCC"]
        - stats9["Average_ZNCC"]
    )

    dx_difference = (
        stats19b["Average_dx"]
        - stats9["Average_dx"]
    )

    dy_difference = (
        stats19b["Average_dy"]
        - stats9["Average_dy"]
    )

    magnitude_difference = (
        stats19b["Average_Magnitude"]
        - stats9["Average_Magnitude"]
    )

    max_difference = (
        stats19b["Maximum_Magnitude"]
        - stats9["Maximum_Magnitude"]
    )


    print()
    print(
        "DIFFERENCE (Stage 19B - Stage 9)"
    )

    print(
        f"Tracked points : "
        f"{point_difference}"
    )

    print(
        f"Average ZNCC   : "
        f"{zncc_difference:.8f}"
    )

    print(
        f"Average dx     : "
        f"{dx_difference:.8f}"
    )

    print(
        f"Average dy     : "
        f"{dy_difference:.8f}"
    )

    print(
        f"Average mag    : "
        f"{magnitude_difference:.8f}"
    )

    print(
        f"Maximum mag    : "
        f"{max_difference:.8f}"
    )


    results.append(
        {
            "Frame_1":
                frame1,

            "Frame_2":
                frame2,

            "Stage9_Tracked_Points":
                stats9["Tracked_Points"],

            "Stage19B_Tracked_Points":
                stats19b["Tracked_Points"],

            "Tracked_Point_Difference":
                point_difference,

            "Stage9_Average_ZNCC":
                stats9["Average_ZNCC"],

            "Stage19B_Average_ZNCC":
                stats19b["Average_ZNCC"],

            "ZNCC_Difference":
                zncc_difference,

            "Stage9_Average_dx":
                stats9["Average_dx"],

            "Stage19B_Average_dx":
                stats19b["Average_dx"],

            "dx_Difference":
                dx_difference,

            "Stage9_Average_dy":
                stats9["Average_dy"],

            "Stage19B_Average_dy":
                stats19b["Average_dy"],

            "dy_Difference":
                dy_difference,

            "Stage9_Average_Magnitude":
                stats9["Average_Magnitude"],

            "Stage19B_Average_Magnitude":
                stats19b["Average_Magnitude"],

            "Average_Magnitude_Difference":
                magnitude_difference,

            "Stage9_Maximum_Magnitude":
                stats9["Maximum_Magnitude"],

            "Stage19B_Maximum_Magnitude":
                stats19b["Maximum_Magnitude"],

            "Maximum_Magnitude_Difference":
                max_difference
        }
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "stage9_vs_stage19B_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 75)
print("STAGE 19C COMPLETE")
print("=" * 75)

print()
print(
    "Comparison saved:"
)

print(
    comparison_path
)

print()

if comparison.empty:

    print(
        "No comparison could be performed."
    )

    print(
        "This is expected if Stage 19B was run on "
        "BGSL rather than BGSL_ROI."
    )

else:

    print(
        "Important: this comparison is only valid if "
        "Stage 19B used the same ROI images as Stage 9."
    )

    print(
        "For the final validation, run Stage 19B using:"
    )

    print()

    print(
        r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"
    )