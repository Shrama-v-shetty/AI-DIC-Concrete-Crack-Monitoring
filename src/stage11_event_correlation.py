import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset"

MASTER_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage10_DIC_Excel_Integration",
    "master_dic_excel.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage11_Event_Correlation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Primary experimental displacement
DISPLACEMENT_COLUMN = "Displacement_ch7_Delta_um"

# Rolling window used to reduce noise
ROLLING_WINDOW = 21

# Percentile used for automatic candidate thresholds
THRESHOLD_PERCENTILE = 99


# ============================================================
# LOAD MASTER DATASET
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11 — DIC + EXPERIMENTAL EVENT CORRELATION")
print("=" * 65)

print("\nLoading Stage 10 master dataset...")

df = pd.read_csv(MASTER_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "Frame_1",
    "Frame_2",
    "Time_1",
    "Time_2",
    "Load_1_kN",
    "Load_2_kN",
    "Load_Delta_kN",
    "Displacement_ch7_1_um",
    "Displacement_ch7_2_um",
    "Displacement_ch7_Delta_um",
    "DIC_Average_dx",
    "DIC_Average_dy",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC",
    "DIC_Tracked_Points",
    "DIC_Processing_Time_sec",
    "Time_Delta_sec"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan
        )


# ============================================================
# SORT BY FRAME
# ============================================================

df = df.sort_values(
    "Frame_2"
).reset_index(drop=True)


# ============================================================
# BASIC DERIVED VALUES
# ============================================================

# Absolute changes are useful for detecting sudden events.

df["Abs_Load_Delta"] = (
    df["Load_Delta_kN"].abs()
)

df["Abs_Displacement_Delta"] = (
    df[DISPLACEMENT_COLUMN].abs()
)

df["DIC_Deformation"] = (
    df["DIC_Average_Magnitude"]
)

df["DIC_Max_Deformation"] = (
    df["DIC_Maximum_Magnitude"]
)


# ============================================================
# ROLLING MEDIAN
# ============================================================

print("\nCalculating rolling-median signals...")

df["Load_Delta_Smoothed"] = (
    df["Load_Delta_kN"]
    .rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
)

df["Displacement_Delta_Smoothed"] = (
    df[DISPLACEMENT_COLUMN]
    .rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
)

df["DIC_Deformation_Smoothed"] = (
    df["DIC_Deformation"]
    .rolling(
        ROLLING_WINDOW,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# AUTOMATIC THRESHOLDS
# ============================================================

load_threshold = df[
    "Abs_Load_Delta"
].quantile(
    THRESHOLD_PERCENTILE / 100
)

displacement_threshold = df[
    "Abs_Displacement_Delta"
].quantile(
    THRESHOLD_PERCENTILE / 100
)

dic_threshold = df[
    "DIC_Deformation"
].quantile(
    THRESHOLD_PERCENTILE / 100
)

dic_max_threshold = df[
    "DIC_Max_Deformation"
].quantile(
    THRESHOLD_PERCENTILE / 100
)


print("\nAutomatic thresholds:")
print(
    f"Load change threshold          : "
    f"{load_threshold:.6f} kN"
)

print(
    f"Ch7 displacement change        : "
    f"{displacement_threshold:.6f} µm"
)

print(
    f"DIC average deformation       : "
    f"{dic_threshold:.6f} px/pair"
)

print(
    f"DIC maximum deformation       : "
    f"{dic_max_threshold:.6f} px/pair"
)


# ============================================================
# NORMALIZED SCORES
# ============================================================

df["Load_Score"] = (
    df["Abs_Load_Delta"]
    / load_threshold
)

df["Displacement_Score"] = (
    df["Abs_Displacement_Delta"]
    / displacement_threshold
)

df["DIC_Score"] = (
    df["DIC_Deformation"]
    / dic_threshold
)

df["DIC_Max_Score"] = (
    df["DIC_Max_Deformation"]
    / dic_max_threshold
)


# ============================================================
# COMBINED EVENT SCORE
# ============================================================

# The score gives greatest importance to experimental
# displacement and DIC deformation, while load change
# provides mechanical context.

df["Event_Score"] = (
    0.30 * df["Load_Score"]
    +
    0.30 * df["Displacement_Score"]
    +
    0.25 * df["DIC_Score"]
    +
    0.15 * df["DIC_Max_Score"]
)


# ============================================================
# LOAD-DECREASE + DISPLACEMENT-INCREASE INDICATOR
# ============================================================

df["Load_Decrease"] = (
    df["Load_Delta_kN"] < 0
)

df["Displacement_Increase"] = (
    df[DISPLACEMENT_COLUMN] > 0
)

df["Load_Drop_With_Displacement_Increase"] = (
    df["Load_Decrease"]
    &
    df["Displacement_Increase"]
)


# ============================================================
# DIC QUALITY
# ============================================================

if "DIC_Average_ZNCC" in df.columns:

    df["DIC_Quality_Flag"] = np.where(
        df["DIC_Average_ZNCC"] >= 0.99,
        "Good",
        "Review"
    )

else:

    df["DIC_Quality_Flag"] = "Unknown"


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def classify_event(row):

    score = row["Event_Score"]

    load_drop = row[
        "Load_Decrease"
    ]

    displacement_increase = row[
        "Displacement_Increase"
    ]

    dic_high = (
        row["DIC_Deformation"]
        >= dic_threshold
    )

    dic_max_high = (
        row["DIC_Max_Deformation"]
        >= dic_max_threshold
    )

    quality = row[
        "DIC_Quality_Flag"
    ]

    # Strong combined candidate
    if (
        score >= 2.5
        and displacement_increase
        and dic_high
        and dic_max_high
    ):

        if load_drop:

            return (
                "Strong deformation candidate "
                "(load drop + displacement increase + DIC)"
            )

        return (
            "Strong deformation candidate "
            "(displacement + DIC)"
        )

    # Mechanical event
    if (
        score >= 2.0
        and load_drop
        and displacement_increase
    ):

        return (
            "Mechanical event candidate "
            "(load drop + displacement increase)"
        )

    # DIC event
    if (
        score >= 2.0
        and dic_high
    ):

        return (
            "DIC deformation candidate"
        )

    # Displacement event
    if (
        score >= 2.0
        and displacement_increase
    ):

        return (
            "Displacement event candidate"
        )

    return "Normal / low significance"


df["Event_Classification"] = (
    df.apply(
        classify_event,
        axis=1
    )
)


# ============================================================
# IMAGE AVAILABILITY
# ============================================================

# Stage 9 contains only image pairs that were actually
# available and successfully tracked.

df["Image_Pair_Available"] = True


# ============================================================
# SORT CANDIDATES
# ============================================================

candidate_df = df[
    df["Event_Classification"]
    != "Normal / low significance"
].copy()

candidate_df = candidate_df.sort_values(
    "Event_Score",
    ascending=False
)


# ============================================================
# SAVE ALL ANALYSIS
# ============================================================

complete_path = os.path.join(
    OUTPUT_DIR,
    "complete_event_analysis.csv"
)

df.to_csv(
    complete_path,
    index=False
)


# ============================================================
# SAVE CANDIDATES
# ============================================================

candidate_path = os.path.join(
    OUTPUT_DIR,
    "event_candidates.csv"
)

candidate_df.to_csv(
    candidate_path,
    index=False
)


# ============================================================
# SAVE TOP 30 EVENTS
# ============================================================

top_events = candidate_df.head(30)

top_path = os.path.join(
    OUTPUT_DIR,
    "top_30_event_candidates.csv"
)

top_events.to_csv(
    top_path,
    index=False
)


# ============================================================
# PRINT TOP EVENTS
# ============================================================

print("\n" + "=" * 65)
print("TOP EVENT CANDIDATES")
print("=" * 65)

display_columns = [
    "Frame_1",
    "Frame_2",
    "Time_2",
    "Load_2_kN",
    "Load_Delta_kN",
    "Displacement_ch7_2_um",
    "Displacement_ch7_Delta_um",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC",
    "Event_Score",
    "Event_Classification"
]

available_display_columns = [
    column
    for column in display_columns
    if column in top_events.columns
]

print(
    top_events[
        available_display_columns
    ].to_string(index=False)
)


# ============================================================
# GRAPH 1 — EVENT SCORE VS FRAME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Frame_2"],
    df["Event_Score"],
    linewidth=1.0
)

plt.xlabel("Frame")
plt.ylabel("Event score")
plt.title(
    "Combined Event Score vs Frame"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "event_score_vs_frame.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(f"\nSaved: {path}")


# ============================================================
# GRAPH 2 — DIC VS CH7 DISPLACEMENT CHANGE
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df[DISPLACEMENT_COLUMN],
    df["DIC_Deformation"],
    s=8,
    alpha=0.6
)

plt.xlabel(
    "Displacement ch 7 change (µm)"
)

plt.ylabel(
    "DIC incremental deformation (px/pair)"
)

plt.title(
    "DIC Deformation vs Channel 7 Displacement Change"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "dic_vs_ch7_displacement_change.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# GRAPH 3 — LOAD CHANGE VS CH7 DISPLACEMENT CHANGE
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Load_Delta_kN"],
    df[DISPLACEMENT_COLUMN],
    s=8,
    alpha=0.6
)

plt.xlabel(
    "Load change (kN)"
)

plt.ylabel(
    "Displacement ch 7 change (µm)"
)

plt.title(
    "Load Change vs Channel 7 Displacement Change"
)

plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    0,
    linewidth=0.8
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "load_change_vs_ch7_displacement_change.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11 COMPLETE")
print("=" * 65)

print(
    f"Total DIC pairs analysed : {len(df)}"
)

print(
    f"Event candidates         : {len(candidate_df)}"
)

print(
    f"Strongest candidates     : "
    f"{min(30, len(candidate_df))}"
)

print("\nPrimary displacement:")
print(
    "Displacement ch 7 (µm)"
)

print("\nOutput directory:")
print(
    OUTPUT_DIR
)

print("\nMain files:")
print(
    complete_path
)

print(
    candidate_path
)

print(
    top_path
)