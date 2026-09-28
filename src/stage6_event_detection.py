import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# PATHS
# ============================================================

EXCEL_FILE = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\experimental_data\combined_image_load_data_0_to_6600.xlsx"
)

OUTPUT_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6_Event_Analysis"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# We will use displacement channel 7 for the first analysis.
# We are NOT claiming this is the final crack channel yet.
DISPLACEMENT_COLUMN = "Displacement ch 7 µm"

LOAD_COLUMN = "Load ch 2 kN"
TIME_COLUMN = "Time_0"
FRAME_COLUMN = "Image Number"


# ============================================================
# READ EXCEL
# ============================================================

print("==========================================")
print("       STAGE 6 - EVENT ANALYSIS")
print("==========================================")

print("\nReading:")
print(EXCEL_FILE)

if not os.path.exists(EXCEL_FILE):

    print("\nERROR: Excel file not found.")
    print(EXCEL_FILE)
    raise SystemExit


df = pd.read_excel(EXCEL_FILE)

print("\nRows:", len(df))


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

numeric_columns = [
    FRAME_COLUMN,
    TIME_COLUMN,
    LOAD_COLUMN,
    "Displacement ch 5 µm",
    "Displacement ch 6 µm",
    "Displacement ch 7 µm",
    "Displacement ch 8 µm",
    "Stress MPa"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Convert infinity to NaN
        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan
        )


# ============================================================
# SORT BY IMAGE NUMBER
# ============================================================

df = df.sort_values(
    FRAME_COLUMN
).reset_index(drop=True)


# ============================================================
# BASIC VALIDATION
# ============================================================

required = [
    FRAME_COLUMN,
    TIME_COLUMN,
    LOAD_COLUMN,
    DISPLACEMENT_COLUMN
]

for column in required:

    if column not in df.columns:

        print(
            "\nERROR: Required column missing:",
            column
        )

        raise SystemExit


# ============================================================
# INTERPOLATE ONLY FOR ANALYSIS
# ============================================================

# We do not modify the original Excel.
# Missing numeric values are interpolated only for calculating
# smooth trends.

df[LOAD_COLUMN] = df[LOAD_COLUMN].interpolate(
    limit_direction="both"
)

df[DISPLACEMENT_COLUMN] = df[
    DISPLACEMENT_COLUMN
].interpolate(
    limit_direction="both"
)

df[TIME_COLUMN] = df[TIME_COLUMN].interpolate(
    limit_direction="both"
)


# ============================================================
# CALCULATE CHANGES
# ============================================================

df["Load_Change"] = df[
    LOAD_COLUMN
].diff()

df["Displacement_Change"] = df[
    DISPLACEMENT_COLUMN
].diff()

df["Time_Change"] = df[
    TIME_COLUMN
].diff()


# ============================================================
# CALCULATE RATES
# ============================================================

safe_time = df["Time_Change"].replace(
    0,
    np.nan
)

df["Load_Rate"] = (
    df["Load_Change"] /
    safe_time
)

df["Displacement_Rate"] = (
    df["Displacement_Change"] /
    safe_time
)


# ============================================================
# SMOOTH DISPLACEMENT
# ============================================================

# Small rolling window to reduce single-row noise.

df["Displacement_Smooth"] = (
    df[DISPLACEMENT_COLUMN]
    .rolling(
        window=21,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# SMOOTH LOAD
# ============================================================

df["Load_Smooth"] = (
    df[LOAD_COLUMN]
    .rolling(
        window=21,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# CALCULATE SMOOTHED CHANGES
# ============================================================

df["Smooth_Displacement_Change"] = (
    df["Displacement_Smooth"].diff()
)

df["Smooth_Load_Change"] = (
    df["Load_Smooth"].diff()
)


# ============================================================
# FIND STRONG DISPLACEMENT CHANGES
# ============================================================

displacement_change_abs = (
    df["Smooth_Displacement_Change"]
    .abs()
)

threshold_displacement = (
    displacement_change_abs
    .quantile(0.99)
)


# ============================================================
# FIND STRONG LOAD CHANGES
# ============================================================

load_change_abs = (
    df["Smooth_Load_Change"]
    .abs()
)

threshold_load = (
    load_change_abs
    .quantile(0.99)
)


print("\n==========================================")
print("       AUTOMATIC THRESHOLDS")
print("==========================================")

print(
    "99th percentile displacement change:",
    threshold_displacement
)

print(
    "99th percentile load change:",
    threshold_load
)


# ============================================================
# IDENTIFY CANDIDATE EVENTS
# ============================================================

# Important:
# This is ONLY a candidate detector.
# It does NOT say that every candidate is a crack.

candidate_mask = (
    displacement_change_abs
    >= threshold_displacement
) | (
    load_change_abs
    >= threshold_load
)


candidate_df = df[
    candidate_mask
].copy()


# ============================================================
# ADD EVENT SCORE
# ============================================================

candidate_df["Displacement_Score"] = (
    displacement_change_abs.loc[
        candidate_df.index
    ] /
    threshold_displacement
)

candidate_df["Load_Score"] = (
    load_change_abs.loc[
        candidate_df.index
    ] /
    threshold_load
)

candidate_df["Event_Score"] = (
    candidate_df["Displacement_Score"] +
    candidate_df["Load_Score"]
)


# ============================================================
# SORT BY EVENT SCORE
# ============================================================

candidate_df = candidate_df.sort_values(
    "Event_Score",
    ascending=False
)


# ============================================================
# SELECT USEFUL COLUMNS
# ============================================================

output_columns = [
    FRAME_COLUMN,
    TIME_COLUMN,
    LOAD_COLUMN,
    DISPLACEMENT_COLUMN,
    "Load_Change",
    "Displacement_Change",
    "Load_Rate",
    "Displacement_Rate",
    "Event_Score"
]

output_columns = [
    column
    for column in output_columns
    if column in candidate_df.columns
]

events = candidate_df[
    output_columns
].copy()


# ============================================================
# SAVE CANDIDATE EVENTS
# ============================================================

events_csv = os.path.join(
    OUTPUT_FOLDER,
    "candidate_events.csv"
)

events.to_csv(
    events_csv,
    index=False
)


# ============================================================
# PRINT TOP EVENTS
# ============================================================

print("\n==========================================")
print("       TOP CANDIDATE EVENTS")
print("==========================================")

print(
    events.head(30).to_string(
        index=False
    )
)


# ============================================================
# LOAD VS TIME
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df[TIME_COLUMN],
    df[LOAD_COLUMN]
)

plt.xlabel("Time")
plt.ylabel("Load (kN)")

plt.title(
    "Load vs Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

load_time_path = os.path.join(
    OUTPUT_FOLDER,
    "load_vs_time.png"
)

plt.savefig(
    load_time_path,
    dpi=200
)

plt.close()


# ============================================================
# DISPLACEMENT VS TIME
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df[TIME_COLUMN],
    df[DISPLACEMENT_COLUMN]
)

plt.xlabel("Time")
plt.ylabel("Displacement (µm)")

plt.title(
    "Displacement vs Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

disp_time_path = os.path.join(
    OUTPUT_FOLDER,
    "displacement_vs_time.png"
)

plt.savefig(
    disp_time_path,
    dpi=200
)

plt.close()


# ============================================================
# LOAD VS DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    df[DISPLACEMENT_COLUMN],
    df[LOAD_COLUMN]
)

plt.xlabel(
    "Displacement (µm)"
)

plt.ylabel(
    "Load (kN)"
)

plt.title(
    "Load vs Displacement"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

load_disp_path = os.path.join(
    OUTPUT_FOLDER,
    "load_vs_displacement.png"
)

plt.savefig(
    load_disp_path,
    dpi=200
)

plt.close()


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_csv = os.path.join(
    OUTPUT_FOLDER,
    "experimental_data_with_analysis.csv"
)

df.to_csv(
    cleaned_csv,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("       STAGE 6 COMPLETE")
print("==========================================")

print("\nCandidate events:")
print(events_csv)

print("\nLoad vs Time:")
print(load_time_path)

print("\nDisplacement vs Time:")
print(disp_time_path)

print("\nLoad vs Displacement:")
print(load_disp_path)

print("\nCleaned analysis data:")
print(cleaned_csv)