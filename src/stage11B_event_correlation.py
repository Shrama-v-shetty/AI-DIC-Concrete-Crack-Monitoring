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
    "Stage11B_Event_Correlation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW = 21

# Primary displacement channel
CH7 = "Displacement_ch7_2_um"

# DIC columns
DIC_AVG = "DIC_Average_Magnitude"
DIC_MAX = "DIC_Maximum_Magnitude"
ZNCC = "DIC_Average_ZNCC"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11B — MULTI-FRAME EVENT CORRELATION")
print("=" * 65)

print("\nLoading Stage 10 master dataset...")

df = pd.read_csv(MASTER_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

numeric_columns = [
    "Frame_1",
    "Frame_2",
    "Time_1",
    "Time_2",
    "Load_1_kN",
    "Load_2_kN",
    "Displacement_ch7_1_um",
    "Displacement_ch7_2_um",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC",
    "DIC_Tracked_Points"
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
# IDENTIFY CONTINUOUS IMAGE SEGMENTS
# ============================================================

print("\nIdentifying continuous image segments...")

segments = []

start_index = 0

for i in range(1, len(df)):

    previous_frame = int(df.loc[i - 1, "Frame_2"])
    current_frame = int(df.loc[i, "Frame_2"])

    # Normally consecutive frames differ by 1.
    # A larger jump indicates a missing-image gap.

    if current_frame != previous_frame + 1:

        segments.append(
            (start_index, i - 1)
        )

        start_index = i

segments.append(
    (start_index, len(df) - 1)
)


print(f"Continuous segments found: {len(segments)}")

for number, (start, end) in enumerate(segments, 1):

    first_frame = int(df.loc[start, "Frame_2"])
    last_frame = int(df.loc[end, "Frame_2"])

    print(
        f"Segment {number}: "
        f"{first_frame} → {last_frame}"
    )


# ============================================================
# CREATE MULTI-FRAME FEATURES
# ============================================================

df["Window_Load_Change_kN"] = np.nan
df["Window_Ch7_Change_um"] = np.nan
df["Window_DIC_Change"] = np.nan
df["Window_DIC_Max_Change"] = np.nan

df["Window_Start_Frame"] = np.nan


for start, end in segments:

    segment_indices = list(
        range(start, end + 1)
    )

    for local_position, index in enumerate(
        segment_indices
    ):

        # Need enough history for the window.
        if local_position < WINDOW:
            continue

        previous_index = segment_indices[
            local_position - WINDOW
        ]

        current_index = index

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        df.loc[
            current_index,
            "Window_Start_Frame"
        ] = df.loc[
            previous_index,
            "Frame_2"
        ]

        # ----------------------------------------------------
        # Load change
        # ----------------------------------------------------

        load_start = df.loc[
            previous_index,
            "Load_2_kN"
        ]

        load_end = df.loc[
            current_index,
            "Load_2_kN"
        ]

        df.loc[
            current_index,
            "Window_Load_Change_kN"
        ] = load_end - load_start

        # ----------------------------------------------------
        # Channel 7 displacement change
        # ----------------------------------------------------

        disp_start = df.loc[
            previous_index,
            "Displacement_ch7_2_um"
        ]

        disp_end = df.loc[
            current_index,
            "Displacement_ch7_2_um"
        ]

        df.loc[
            current_index,
            "Window_Ch7_Change_um"
        ] = disp_end - disp_start

        # ----------------------------------------------------
        # DIC change
        # ----------------------------------------------------

        dic_values = df.loc[
            previous_index + 1:
            current_index,
            DIC_AVG
        ]

        dic_max_values = df.loc[
            previous_index + 1:
            current_index,
            DIC_MAX
        ]

        if len(dic_values) > 0:

            df.loc[
                current_index,
                "Window_DIC_Change"
            ] = dic_values.mean()

            df.loc[
                current_index,
                "Window_DIC_Max_Change"
            ] = dic_max_values.max()


# ============================================================
# ABSOLUTE VALUES
# ============================================================

df["Abs_Window_Load_Change"] = (
    df["Window_Load_Change_kN"].abs()
)

df["Abs_Window_Ch7_Change"] = (
    df["Window_Ch7_Change_um"].abs()
)


# ============================================================
# AUTOMATIC THRESHOLDS
# ============================================================

valid = df[
    df["Window_Ch7_Change_um"].notna()
].copy()


load_threshold = valid[
    "Abs_Window_Load_Change"
].quantile(0.95)

ch7_threshold = valid[
    "Abs_Window_Ch7_Change"
].quantile(0.95)

dic_threshold = valid[
    "Window_DIC_Change"
].quantile(0.95)

dic_max_threshold = valid[
    "Window_DIC_Max_Change"
].quantile(0.95)


print("\n95th percentile thresholds:")

print(
    f"Load change       : "
    f"{load_threshold:.6f} kN"
)

print(
    f"Ch7 displacement  : "
    f"{ch7_threshold:.6f} µm"
)

print(
    f"DIC average       : "
    f"{dic_threshold:.6f} px/pair"
)

print(
    f"DIC maximum       : "
    f"{dic_max_threshold:.6f} px/pair"
)


# ============================================================
# NORMALIZED SCORES
# ============================================================

df["Load_Score"] = (
    df["Abs_Window_Load_Change"]
    / load_threshold
)

df["Ch7_Score"] = (
    df["Abs_Window_Ch7_Change"]
    / ch7_threshold
)

df["DIC_Score"] = (
    df["Window_DIC_Change"]
    / dic_threshold
)

df["DIC_Max_Score"] = (
    df["Window_DIC_Max_Change"]
    / dic_max_threshold
)


# ============================================================
# CORRELATED BEHAVIOUR
# ============================================================

df["Load_Decrease"] = (
    df["Window_Load_Change_kN"] < 0
)

df["Ch7_Increase"] = (
    df["Window_Ch7_Change_um"] > 0
)

df["Load_Drop_Ch7_Increase"] = (
    df["Load_Decrease"]
    &
    df["Ch7_Increase"]
)


# ============================================================
# DIC SUPPORT
# ============================================================

df["DIC_Support"] = (
    (df["Window_DIC_Change"] >= dic_threshold)
    |
    (df["Window_DIC_Max_Change"] >= dic_max_threshold)
)


# ============================================================
# CORRELATION SCORE
# ============================================================

# Experimental behaviour gets the greatest weight.
# DIC provides supporting evidence.

df["Correlation_Score"] = (
    0.40 * df["Ch7_Score"]
    +
    0.30 * df["Load_Score"]
    +
    0.20 * df["DIC_Score"]
    +
    0.10 * df["DIC_Max_Score"]
)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(row):

    score = row["Correlation_Score"]

    ch7_change = row[
        "Window_Ch7_Change_um"
    ]

    load_change = row[
        "Window_Load_Change_kN"
    ]

    dic_support = row[
        "DIC_Support"
    ]

    # --------------------------------------------------------
    # Strong candidate:
    # significant displacement + DIC support
    # --------------------------------------------------------

    if (
        score >= 2.0
        and abs(ch7_change) >= ch7_threshold
        and dic_support
    ):

        if (
            load_change < 0
            and ch7_change > 0
        ):

            return (
                "Strong correlated deformation candidate "
                "(load drop + Ch7 increase + DIC)"
            )

        return (
            "Strong correlated deformation candidate "
            "(Ch7 + DIC)"
        )

    # --------------------------------------------------------
    # Mechanical candidate
    # --------------------------------------------------------

    if (
        abs(ch7_change) >= ch7_threshold
        and load_change < 0
    ):

        return (
            "Mechanical deformation candidate "
            "(load drop + Ch7 increase)"
        )

    # --------------------------------------------------------
    # Displacement candidate
    # --------------------------------------------------------

    if (
        abs(ch7_change) >= ch7_threshold
    ):

        return (
            "Ch7 displacement candidate"
        )

    # --------------------------------------------------------
    # DIC-only anomaly
    # --------------------------------------------------------

    if dic_support:

        return (
            "DIC deformation anomaly — review"
        )

    return "Normal / low significance"


df["Event_Classification"] = (
    df.apply(
        classify,
        axis=1
    )
)


# ============================================================
# REMOVE INVALID WINDOW ROWS
# ============================================================

analysis_df = df[
    df["Window_Ch7_Change_um"].notna()
].copy()


# ============================================================
# EVENT CANDIDATES
# ============================================================

candidate_df = analysis_df[
    analysis_df["Event_Classification"]
    != "Normal / low significance"
].copy()

candidate_df = candidate_df.sort_values(
    "Correlation_Score",
    ascending=False
)


# ============================================================
# STRONG CANDIDATES ONLY
# ============================================================

strong_df = analysis_df[
    analysis_df["Event_Classification"].str.startswith(
        "Strong correlated"
    )
].copy()

strong_df = strong_df.sort_values(
    "Correlation_Score",
    ascending=False
)


# ============================================================
# SAVE FILES
# ============================================================

complete_path = os.path.join(
    OUTPUT_DIR,
    "complete_event_analysis_21frame.csv"
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "event_candidates_21frame.csv"
)

strong_path = os.path.join(
    OUTPUT_DIR,
    "strong_correlated_events.csv"
)

top_path = os.path.join(
    OUTPUT_DIR,
    "top_30_event_candidates_21frame.csv"
)


df.to_csv(
    complete_path,
    index=False
)

candidate_df.to_csv(
    candidate_path,
    index=False
)

strong_df.to_csv(
    strong_path,
    index=False
)

candidate_df.head(30).to_csv(
    top_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TOP MULTI-FRAME EVENT CANDIDATES")
print("=" * 65)

display_columns = [
    "Window_Start_Frame",
    "Frame_2",
    "Time_2",
    "Load_2_kN",
    "Window_Load_Change_kN",
    "Displacement_ch7_2_um",
    "Window_Ch7_Change_um",
    "Window_DIC_Change",
    "Window_DIC_Max_Change",
    "DIC_Average_ZNCC",
    "Correlation_Score",
    "Event_Classification"
]

display_columns = [
    c for c in display_columns
    if c in candidate_df.columns
]

if len(candidate_df) > 0:

    print(
        candidate_df.head(30)[
            display_columns
        ].to_string(index=False)
    )

else:

    print("No event candidates found.")


# ============================================================
# PRINT STRONG EVENTS
# ============================================================

print("\n" + "=" * 65)
print("STRONG CORRELATED EVENTS")
print("=" * 65)

if len(strong_df) > 0:

    print(
        strong_df.head(30)[
            display_columns
        ].to_string(index=False)
    )

else:

    print(
        "No strong correlated events "
        "were detected using the current thresholds."
    )


# ============================================================
# GRAPH 1 — CORRELATION SCORE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    analysis_df["Frame_2"],
    analysis_df["Correlation_Score"],
    linewidth=1.0
)

plt.xlabel("Frame")
plt.ylabel("Correlation score")
plt.title(
    "Multi-Frame Correlation Score vs Frame"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "correlation_score_vs_frame.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 2 — CH7 WINDOW CHANGE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    analysis_df["Frame_2"],
    analysis_df["Window_Ch7_Change_um"],
    linewidth=1.0
)

plt.xlabel("Frame")
plt.ylabel(
    "21-frame Ch7 displacement change (µm)"
)

plt.title(
    "21-Frame Channel 7 Displacement Change"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "ch7_21frame_change_vs_frame.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 3 — LOAD WINDOW CHANGE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    analysis_df["Frame_2"],
    analysis_df["Window_Load_Change_kN"],
    linewidth=1.0
)

plt.xlabel("Frame")
plt.ylabel(
    "21-frame load change (kN)"
)

plt.title(
    "21-Frame Load Change"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "load_21frame_change_vs_frame.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 4 — DIC WINDOW DEFORMATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    analysis_df["Frame_2"],
    analysis_df["Window_DIC_Change"],
    linewidth=1.0
)

plt.xlabel("Frame")
plt.ylabel(
    "Mean DIC incremental deformation (px/pair)"
)

plt.title(
    "21-Frame DIC Deformation"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "dic_21frame_deformation_vs_frame.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STAGE 11B COMPLETE")
print("=" * 65)

print(
    f"Total DIC pairs analysed : {len(df)}"
)

print(
    f"Valid 21-frame windows   : {len(analysis_df)}"
)

print(
    f"Event candidates         : {len(candidate_df)}"
)

print(
    f"Strong correlated events : {len(strong_df)}"
)

print("\nPrimary displacement:")
print(
    "Displacement ch 7 (µm)"
)

print("\nOutput directory:")
print(
    OUTPUT_DIR
)

print("\nMain result:")
print(
    strong_path
)

print("\nTop candidates:")
print(
    top_path
)