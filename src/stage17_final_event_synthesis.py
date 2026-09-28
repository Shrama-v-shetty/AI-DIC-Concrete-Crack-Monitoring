import os
import pandas as pd
import numpy as np


# ============================================================
# STAGE 17 - FINAL EVENT SYNTHESIS
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

STAGE14_DIR = os.path.join(
    BASE_DIR,
    "Stage14_Strong_Event_Verification"
)

STAGE15_DIR = os.path.join(
    BASE_DIR,
    "Stage15_Spatial_Event_Validation"
)

STAGE16_DIR = os.path.join(
    BASE_DIR,
    "Stage16_Spatial_Localization"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage17_Final_Event_Synthesis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INPUTS
# ============================================================

EVENTS_FILE = os.path.join(
    STAGE14_DIR,
    "verified_strong_events.csv"
)

SPATIAL_FILE = os.path.join(
    STAGE15_DIR,
    "spatial_event_validation_summary.csv"
)

LOCALIZATION_FILE = os.path.join(
    STAGE16_DIR,
    "spatial_localization_classification.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STAGE 17 - FINAL EVENT SYNTHESIS")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    EVENTS_FILE,
    SPATIAL_FILE,
    LOCALIZATION_FILE
]:

    print()
    print("Checking:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

events = pd.read_csv(
    EVENTS_FILE
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

localization = pd.read_csv(
    LOCALIZATION_FILE
)


print()
print(
    f"Strong events loaded : {len(events)}"
)

print(
    f"Spatial records      : {len(spatial)}"
)

print(
    f"Localization records : {len(localization)}"
)


# ============================================================
# MERGE USING FRAME NUMBERS
# ============================================================

merge_columns = [
    "Frame_1",
    "Frame_2"
]

spatial_columns = [
    "Frame_1",
    "Frame_2",
    "Principal_Strain_Mean",
    "Principal_Strain_P95",
    "Principal_Strain_P99",
    "Principal_Strain_Max",
    "High_Strain_Cells",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_X_Min",
    "High_Strain_X_Max",
    "High_Strain_Y_Min",
    "High_Strain_Y_Max",
    "Validation_Image"
]

spatial_columns = [
    c
    for c in spatial_columns
    if c in spatial.columns
]

localization_columns = [
    "Frame_1",
    "Frame_2",
    "High_Strain_Width",
    "High_Strain_Height",
    "Centroid_Proximity_Score",
    "Spatial_Compactness_Score",
    "Localization_Score",
    "Spatial_Localization_Category"
]

localization_columns = [
    c
    for c in localization_columns
    if c in localization.columns
]


spatial_subset = spatial[
    spatial_columns
].copy()

localization_subset = localization[
    localization_columns
].copy()


final = events.merge(
    spatial_subset,
    on=merge_columns,
    how="left",
    suffixes=("", "_Spatial")
)

final = final.merge(
    localization_subset,
    on=merge_columns,
    how="left",
    suffixes=("", "_Localization")
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "Load_Delta_kN",
    "Primary_Displacement_Delta_um",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC",
    "Principal_Strain_Mean",
    "Principal_Strain_P95",
    "Principal_Strain_P99",
    "Principal_Strain_Max",
    "High_Strain_Cells",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_Width",
    "High_Strain_Height",
    "Centroid_Proximity_Score",
    "Spatial_Compactness_Score",
    "Localization_Score"
]

for column in numeric_columns:

    if column in final.columns:

        final[column] = pd.to_numeric(
            final[column],
            errors="coerce"
        )


# ============================================================
# FINAL INTERPRETATION CATEGORY
# ============================================================

def final_category(row):

    spatial_category = row.get(
        "Spatial_Localization_Category",
        ""
    )

    if spatial_category == (
        "Localized central deformation"
    ):

        return "High-confidence localized deformation"

    if spatial_category == (
        "Central deformation but spatially broad"
    ):

        return "Central but spatially broad deformation"

    if spatial_category == (
        "Compact deformation outside target region"
    ):

        return "Localized but outside expected central region"

    return "Broad/global deformation"


final[
    "Final_Interpretation"
] = final.apply(
    final_category,
    axis=1
)


# ============================================================
# MECHANICAL RESPONSE TYPE
# ============================================================

def mechanical_response(row):

    load_delta = row[
        "Load_Delta_kN"
    ]

    disp_delta = row[
        "Primary_Displacement_Delta_um"
    ]

    if (
        load_delta < 0
        and disp_delta > 0
    ):

        return (
            "Load drop + displacement rise"
        )

    if (
        load_delta > 0
        and disp_delta > 0
    ):

        return (
            "Load rise + displacement rise"
        )

    if (
        load_delta < 0
        and disp_delta < 0
    ):

        return (
            "Load drop + displacement drop"
        )

    return "Mixed response"


final[
    "Mechanical_Response"
] = final.apply(
    mechanical_response,
    axis=1
)


# ============================================================
# DIC QUALITY INTERPRETATION
# ============================================================

def quality_category(zncc):

    if not np.isfinite(zncc):

        return "Unknown"

    if zncc >= 0.995:

        return "Very good"

    if zncc >= 0.985:

        return "Good"

    if zncc >= 0.98:

        return "Acceptable"

    return "Low"


final[
    "DIC_Quality_Final"
] = final[
    "DIC_Average_ZNCC"
].apply(
    quality_category
)


# ============================================================
# FINAL PRIORITY
# ============================================================

priority_map = {
    "High-confidence localized deformation": 4,
    "Central but spatially broad deformation": 3,
    "Localized but outside expected central region": 2,
    "Broad/global deformation": 1
}

final[
    "Final_Priority"
] = final[
    "Final_Interpretation"
].map(
    priority_map
).fillna(0)


# ============================================================
# ADD SIMPLE EVIDENCE FLAGS
# ============================================================

final[
    "Load_Drop"
] = (
    final["Load_Delta_kN"] < 0
)

final[
    "Displacement_Rise"
] = (
    final[
        "Primary_Displacement_Delta_um"
    ] > 0
)

final[
    "Localized"
] = (
    final[
        "Final_Interpretation"
    ]
    == "High-confidence localized deformation"
)


# ============================================================
# SORT
# ============================================================

final = final.sort_values(
    [
        "Final_Priority",
        "Localization_Score",
        "Frame_1"
    ],
    ascending=[
        False,
        False,
        True
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL RANK
# ============================================================

final[
    "Final_Event_Rank"
] = np.arange(
    1,
    len(final) + 1
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 75)
print("FINAL EVENT SYNTHESIS")
print("=" * 75)

display_columns = [
    "Final_Event_Rank",
    "Frame_1",
    "Frame_2",
    "Time_1",
    "Time_2",
    "Load_Delta_kN",
    "Primary_Displacement_Delta_um",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC",
    "Principal_Strain_P99",
    "Principal_Strain_Max",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "High_Strain_Width",
    "High_Strain_Height",
    "Localization_Score",
    "Mechanical_Response",
    "DIC_Quality_Final",
    "Final_Interpretation"
]

display_columns = [
    c
    for c in display_columns
    if c in final.columns
]

print()

print(
    final[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

category_counts = (
    final[
        "Final_Interpretation"
    ]
    .value_counts()
    .rename_axis(
        "Final_Interpretation"
    )
    .reset_index(
        name="Count"
    )
)

print()
print("FINAL CATEGORY COUNTS")
print()

print(
    category_counts.to_string(
        index=False
    )
)


# ============================================================
# SAVE FINAL EVENT DATASET
# ============================================================

final_csv = os.path.join(
    OUTPUT_DIR,
    "final_event_synthesis.csv"
)

final.to_csv(
    final_csv,
    index=False
)


# ============================================================
# SAVE LOCALIZED EVENTS
# ============================================================

localized = final[
    final["Localized"]
].copy()

localized_csv = os.path.join(
    OUTPUT_DIR,
    "high_confidence_localized_events.csv"
)

localized.to_csv(
    localized_csv,
    index=False
)


# ============================================================
# SAVE CATEGORY COUNTS
# ============================================================

category_csv = os.path.join(
    OUTPUT_DIR,
    "final_event_category_counts.csv"
)

category_counts.to_csv(
    category_csv,
    index=False
)


# ============================================================
# SAVE HUMAN-READABLE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "final_event_synthesis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "FINAL EVENT SYNTHESIS REPORT\n"
    )

    report.write(
        "============================\n\n"
    )

    report.write(
        f"Strong correlated events analysed: "
        f"{len(final)}\n\n"
    )

    report.write(
        "CATEGORY COUNTS\n"
    )

    for _, row in category_counts.iterrows():

        report.write(
            f"{row['Final_Interpretation']}: "
            f"{int(row['Count'])}\n"
        )

    report.write(
        "\nEVENT DETAILS\n\n"
    )

    for _, row in final.iterrows():

        report.write(
            f"Event rank {int(row['Final_Event_Rank'])}\n"
        )

        report.write(
            f"Frames: "
            f"{int(row['Frame_1'])} -> "
            f"{int(row['Frame_2'])}\n"
        )

        report.write(
            f"Time: "
            f"{row['Time_1']:.3f} -> "
            f"{row['Time_2']:.3f} sec\n"
        )

        report.write(
            f"Load change: "
            f"{row['Load_Delta_kN']:.6f} kN\n"
        )

        report.write(
            f"Ch7 displacement change: "
            f"{row['Primary_Displacement_Delta_um']:.6f} µm\n"
        )

        report.write(
            f"Average DIC deformation: "
            f"{row['DIC_Average_Magnitude']:.6f} px\n"
        )

        report.write(
            f"Maximum DIC deformation: "
            f"{row['DIC_Maximum_Magnitude']:.6f} px\n"
        )

        report.write(
            f"P99 principal strain: "
            f"{row['Principal_Strain_P99']:.6f}\n"
        )

        report.write(
            f"Maximum principal strain: "
            f"{row['Principal_Strain_Max']:.6f}\n"
        )

        report.write(
            f"High-strain centroid: "
            f"({row['High_Strain_Centroid_X']:.1f}, "
            f"{row['High_Strain_Centroid_Y']:.1f})\n"
        )

        report.write(
            f"High-strain width: "
            f"{row['High_Strain_Width']:.1f} px\n"
        )

        report.write(
            f"High-strain height: "
            f"{row['High_Strain_Height']:.1f} px\n"
        )

        report.write(
            f"Localization score: "
            f"{row['Localization_Score']:.4f}\n"
        )

        report.write(
            f"Mechanical response: "
            f"{row['Mechanical_Response']}\n"
        )

        report.write(
            f"DIC quality: "
            f"{row['DIC_Quality_Final']}\n"
        )

        report.write(
            f"Final interpretation: "
            f"{row['Final_Interpretation']}\n\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STAGE 17 COMPLETE")
print("=" * 75)

print()
print(
    f"Events analysed : {len(final)}"
)

print(
    f"Localized events: {len(localized)}"
)

print()
print(
    "Final event dataset:"
)

print(
    final_csv
)

print()
print(
    "High-confidence localized events:"
)

print(
    localized_csv
)

print()
print(
    "Category counts:"
)

print(
    category_csv
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
    "Important: localized deformation is not automatically "
    "labelled as a confirmed crack."
)

print()
print(
    "Next stage: build the final user-facing analysis software."
)