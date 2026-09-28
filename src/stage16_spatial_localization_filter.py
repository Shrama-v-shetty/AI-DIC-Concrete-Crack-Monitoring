import os
import numpy as np
import pandas as pd


# ============================================================
# STAGE 16 - SPATIAL LOCALIZATION FILTER
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage15_Spatial_Event_Validation"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage16_Spatial_Localization"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

INPUT_CSV = os.path.join(
    INPUT_DIR,
    "spatial_event_validation_summary.csv"
)


# ============================================================
# EXPECTED CENTRAL DEFORMATION REGION
# ============================================================
#
# Based on the previously validated Stage 12 progression,
# the significant deformation region is expected around
# the central part of the beam.
#
# These are spatial filtering limits, not crack boundaries.
# ============================================================

TARGET_X_MIN = 850
TARGET_X_MAX = 1150

TARGET_Y_MIN = 150
TARGET_Y_MAX = 450


# ============================================================
# LOCALIZATION LIMITS
# ============================================================
#
# A localized event should have a reasonably compact
# high-strain bounding box.
#
# These values are used to distinguish broad/global
# deformation from concentrated deformation.
# ============================================================

MAX_LOCALIZED_WIDTH = 300
MAX_LOCALIZED_HEIGHT = 350

MIN_LOCALIZATION_SCORE = 0.20


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STAGE 16 - SPATIAL LOCALIZATION FILTER")
print("=" * 75)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_CSV):

    raise FileNotFoundError(
        f"Stage 15 summary not found:\n{INPUT_CSV}"
    )

print()
print("Input file:")
print(INPUT_CSV)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print()
print(
    f"Events loaded: {len(df)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Event_Number",
    "Frame_1",
    "Frame_2",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_X_Min",
    "High_Strain_X_Max",
    "High_Strain_Y_Min",
    "High_Strain_Y_Max"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Event_Number",
    "Frame_1",
    "Frame_2",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_X_Min",
    "High_Strain_X_Max",
    "High_Strain_Y_Min",
    "High_Strain_Y_Max"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CALCULATE BOUNDING BOX SIZE
# ============================================================

df["High_Strain_Width"] = (
    df["High_Strain_X_Max"]
    - df["High_Strain_X_Min"]
)

df["High_Strain_Height"] = (
    df["High_Strain_Y_Max"]
    - df["High_Strain_Y_Min"]
)


# ============================================================
# TARGET-REGION CHECK
# ============================================================

df["Centroid_In_Target_X"] = (
    (df["High_Strain_Centroid_X"] >= TARGET_X_MIN)
    &
    (df["High_Strain_Centroid_X"] <= TARGET_X_MAX)
)

df["Centroid_In_Target_Y"] = (
    (df["High_Strain_Centroid_Y"] >= TARGET_Y_MIN)
    &
    (df["High_Strain_Centroid_Y"] <= TARGET_Y_MAX)
)

df["Centroid_In_Target_Region"] = (
    df["Centroid_In_Target_X"]
    &
    df["Centroid_In_Target_Y"]
)


# ============================================================
# BOUNDING BOX LOCALIZATION
# ============================================================

df["Compact_Width"] = (
    df["High_Strain_Width"]
    <= MAX_LOCALIZED_WIDTH
)

df["Compact_Height"] = (
    df["High_Strain_Height"]
    <= MAX_LOCALIZED_HEIGHT
)

df["Compact_High_Strain_Region"] = (
    df["Compact_Width"]
    &
    df["Compact_Height"]
)


# ============================================================
# CENTROID DISTANCE FROM TARGET CENTER
# ============================================================

target_center_x = (
    TARGET_X_MIN
    + TARGET_X_MAX
) / 2.0

target_center_y = (
    TARGET_Y_MIN
    + TARGET_Y_MAX
) / 2.0

df["Centroid_Distance"] = np.sqrt(
    (
        df["High_Strain_Centroid_X"]
        - target_center_x
    ) ** 2
    +
    (
        df["High_Strain_Centroid_Y"]
        - target_center_y
    ) ** 2
)


# ============================================================
# TARGET REGION DIAGONAL
# ============================================================

target_diagonal = np.sqrt(
    (
        TARGET_X_MAX
        - TARGET_X_MIN
    ) ** 2
    +
    (
        TARGET_Y_MAX
        - TARGET_Y_MIN
    ) ** 2
)


# ============================================================
# CENTROID PROXIMITY SCORE
# ============================================================

df["Centroid_Proximity_Score"] = (
    1.0
    -
    (
        df["Centroid_Distance"]
        / target_diagonal
    )
)

df["Centroid_Proximity_Score"] = (
    df["Centroid_Proximity_Score"]
    .clip(
        lower=0.0,
        upper=1.0
    )
)


# ============================================================
# COMPACTNESS SCORE
# ============================================================
#
# Smaller bounding boxes receive a larger score.
#
# This is a spatial compactness indicator only.
# ============================================================

normalised_width = (
    df["High_Strain_Width"]
    / MAX_LOCALIZED_WIDTH
)

normalised_height = (
    df["High_Strain_Height"]
    / MAX_LOCALIZED_HEIGHT
)

compactness = 1.0 / (
    1.0
    + normalised_width
    + normalised_height
)

df["Spatial_Compactness_Score"] = (
    compactness
    .clip(
        lower=0.0,
        upper=1.0
    )
)


# ============================================================
# OVERALL LOCALIZATION SCORE
# ============================================================
#
# Equal contribution from:
#
# 1. Target-region proximity
# 2. Spatial compactness
# ============================================================

df["Localization_Score"] = (
    0.5
    * df["Centroid_Proximity_Score"]
    +
    0.5
    * df["Spatial_Compactness_Score"]
)


# ============================================================
# LOCALIZATION CLASSIFICATION
# ============================================================

def classify_localization(row):

    if (
        row["Centroid_In_Target_Region"]
        and
        row["Compact_High_Strain_Region"]
        and
        row["Localization_Score"]
        >= MIN_LOCALIZATION_SCORE
    ):

        return (
            "Localized central deformation"
        )

    if (
        row["Centroid_In_Target_Region"]
        and
        row["Localization_Score"]
        >= MIN_LOCALIZATION_SCORE
    ):

        return (
            "Central deformation but spatially broad"
        )

    if (
        row["Compact_High_Strain_Region"]
        and
        row["Centroid_Proximity_Score"]
        < 0.5
    ):

        return (
            "Compact deformation outside target region"
        )

    return (
        "Broad/global deformation"
    )


df["Spatial_Localization_Category"] = (
    df.apply(
        classify_localization,
        axis=1
    )
)


# ============================================================
# HIGH-CONFIDENCE EVENTS
# ============================================================

localized = df[
    df[
        "Spatial_Localization_Category"
    ]
    == "Localized central deformation"
].copy()


# ============================================================
# SORT LOCALIZED EVENTS
# ============================================================

localized = localized.sort_values(
    [
        "Localization_Score",
        "Frame_1"
    ],
    ascending=[
        False,
        True
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL RANK
# ============================================================

localized[
    "Localization_Rank"
] = np.arange(
    1,
    len(localized) + 1
)


# ============================================================
# PRINT ALL EVENTS
# ============================================================

print()
print("=" * 75)
print("EVENT SPATIAL CLASSIFICATION")
print("=" * 75)

display_columns = [
    "Event_Number",
    "Frame_1",
    "Frame_2",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_Width",
    "High_Strain_Height",
    "Centroid_Proximity_Score",
    "Spatial_Compactness_Score",
    "Localization_Score",
    "Spatial_Localization_Category"
]

print()

print(
    df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# PRINT LOCALIZED EVENTS
# ============================================================

print()
print("=" * 75)
print("LOCALIZED CENTRAL DEFORMATION EVENTS")
print("=" * 75)

if len(localized) == 0:

    print(
        "No events satisfied the spatial localization criteria."
    )

else:

    print()

    print(
        localized[
            display_columns
            + ["Localization_Rank"]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE FULL CLASSIFICATION
# ============================================================

full_output = os.path.join(
    OUTPUT_DIR,
    "spatial_localization_classification.csv"
)

df.to_csv(
    full_output,
    index=False
)


# ============================================================
# SAVE LOCALIZED EVENTS
# ============================================================

localized_output = os.path.join(
    OUTPUT_DIR,
    "localized_deformation_events.csv"
)

localized.to_csv(
    localized_output,
    index=False
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

category_counts = (
    df[
        "Spatial_Localization_Category"
    ]
    .value_counts()
    .rename_axis(
        "Spatial_Localization_Category"
    )
    .reset_index(
        name="Count"
    )
)

category_output = os.path.join(
    OUTPUT_DIR,
    "spatial_localization_category_counts.csv"
)

category_counts.to_csv(
    category_output,
    index=False
)


# ============================================================
# SUMMARY REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "stage16_localization_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "STAGE 16 - SPATIAL LOCALIZATION REPORT\n"
    )

    report.write(
        "======================================\n\n"
    )

    report.write(
        f"Total events analysed: {len(df)}\n"
    )

    report.write(
        f"Localized central events: {len(localized)}\n\n"
    )

    report.write(
        "TARGET REGION\n"
    )

    report.write(
        f"X: {TARGET_X_MIN} -> {TARGET_X_MAX}\n"
    )

    report.write(
        f"Y: {TARGET_Y_MIN} -> {TARGET_Y_MAX}\n\n"
    )

    report.write(
        "LOCALIZATION LIMITS\n"
    )

    report.write(
        f"Maximum width: "
        f"{MAX_LOCALIZED_WIDTH} pixels\n"
    )

    report.write(
        f"Maximum height: "
        f"{MAX_LOCALIZED_HEIGHT} pixels\n"
    )

    report.write(
        f"Minimum localization score: "
        f"{MIN_LOCALIZATION_SCORE:.2f}\n\n"
    )

    report.write(
        "CATEGORY COUNTS\n"
    )

    for _, row in category_counts.iterrows():

        report.write(
            f"{row['Spatial_Localization_Category']}: "
            f"{int(row['Count'])}\n"
        )

    report.write(
        "\nLOCALIZED EVENTS\n"
    )

    for _, row in localized.iterrows():

        report.write(
            f"Rank {int(row['Localization_Rank'])}: "
            f"{int(row['Frame_1'])} -> "
            f"{int(row['Frame_2'])}, "
            f"centroid "
            f"({row['High_Strain_Centroid_X']:.1f}, "
            f"{row['High_Strain_Centroid_Y']:.1f}), "
            f"localization score "
            f"{row['Localization_Score']:.4f}\n"
        )

    report.write(
        "\nINTERPRETATION\n"
    )

    report.write(
        "Spatial localization is used here to distinguish "
        "concentrated deformation from broad/global "
        "deformation. A localized event should still not "
        "be labelled as a confirmed crack without image "
        "or additional crack-specific validation.\n"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 75)
print("STAGE 16 COMPLETE")
print("=" * 75)

print()
print(
    f"Total events analysed : {len(df)}"
)

print(
    f"Localized events      : {len(localized)}"
)

print()
print(
    "Full classification:"
)

print(
    full_output
)

print()
print(
    "Localized events:"
)

print(
    localized_output
)

print()
print(
    "Category counts:"
)

print(
    category_output
)

print()
print(
    "Report:"
)

print(
    report_path
)

print()
print(
    "Next step: review the localized events against their "
    "Stage 15 validation images."
)