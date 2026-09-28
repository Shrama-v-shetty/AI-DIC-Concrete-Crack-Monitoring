import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\shrama\Desktop\Dam updated dataset"

EXCEL_FILE = os.path.join(
    ROOT,
    "experimental_data",
    "combined_image_load_data_0_to_6600.xlsx"
)

DIC_SUMMARY_FILE = os.path.join(
    ROOT,
    "Dataset",
    "Stage6F_Crack_Progression",
    "crack_progression_dic_summary.csv"
)

# If Stage6F summary is not there, use Stage6F strain summary
STRAIN_SUMMARY_FILE = os.path.join(
    ROOT,
    "Dataset",
    "Stage6F_Strain_Analysis",
    "stage6F_strain_band_summary.csv"
)

STAGE6C_DIR = os.path.join(
    ROOT,
    "Dataset",
    "Stage6C_Crack_Progression"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "Dataset",
    "BGSL_ROI"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "Dataset",
    "Stage7_Final_Correlation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

PRIMARY_DISPLACEMENT = (
    "Displacement ch 7 µm"
)

# Rolling window used to reduce one-frame noise
ROLLING_WINDOW = 21

# Event scoring thresholds
DISP_PERCENTILE = 99
LOAD_PERCENTILE = 99


# ============================================================
# LOAD EXCEL
# ============================================================

print("==============================================")
print(" STAGE 7 - FINAL EVENT CORRELATION")
print("==============================================")

print("\nLoading experimental Excel...")

df = pd.read_excel(
    EXCEL_FILE
)

print(
    "Rows:",
    len(df)
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "Image Number",
    "Time_0",
    "Load ch 2 kN",
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

# Replace infinity
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.sort_values(
    "Image Number"
).reset_index(
    drop=True
)


# ============================================================
# INTERPOLATED ANALYSIS COPY
# ============================================================

analysis = df.copy()

analysis[
    "Load_Analysis"
] = analysis[
    "Load ch 2 kN"
].interpolate(
    limit_direction="both"
)

analysis[
    "Displacement_Analysis"
] = analysis[
    PRIMARY_DISPLACEMENT
].interpolate(
    limit_direction="both"
)


# ============================================================
# CHANGE OVER FRAMES
# ============================================================

analysis[
    "Load_Change"
] = analysis[
    "Load_Analysis"
].diff()

analysis[
    "Displacement_Change"
] = analysis[
    "Displacement_Analysis"
].diff()


# ============================================================
# ROLLING MEDIAN
# ============================================================

analysis[
    "Load_Smooth"
] = analysis[
    "Load_Analysis"
].rolling(
    ROLLING_WINDOW,
    center=True,
    min_periods=1
).median()

analysis[
    "Displacement_Smooth"
] = analysis[
    "Displacement_Analysis"
].rolling(
    ROLLING_WINDOW,
    center=True,
    min_periods=1
).median()


analysis[
    "Smooth_Load_Change"
] = analysis[
    "Load_Smooth"
].diff()

analysis[
    "Smooth_Displacement_Change"
] = analysis[
    "Displacement_Smooth"
].diff()


# ============================================================
# AUTOMATIC THRESHOLDS
# ============================================================

load_change_abs = np.abs(
    analysis[
        "Smooth_Load_Change"
    ].dropna()
)

disp_change_abs = np.abs(
    analysis[
        "Smooth_Displacement_Change"
    ].dropna()
)

load_threshold = np.percentile(
    load_change_abs,
    LOAD_PERCENTILE
)

disp_threshold = np.percentile(
    disp_change_abs,
    DISP_PERCENTILE
)

print("\nAUTOMATIC THRESHOLDS")

print(
    f"Load change threshold: "
    f"{load_threshold:.4f} kN"
)

print(
    f"Displacement change threshold: "
    f"{disp_threshold:.4f} µm"
)


# ============================================================
# RATES
# ============================================================

analysis[
    "Time_Change"
] = analysis[
    "Time_0"
].diff()

analysis[
    "Load_Rate"
] = (
    analysis["Smooth_Load_Change"]
    /
    analysis["Time_Change"]
)

analysis[
    "Displacement_Rate"
] = (
    analysis["Smooth_Displacement_Change"]
    /
    analysis["Time_Change"]
)


# ============================================================
# EVENT SCORE
#
# Important:
#
# We give stronger score when:
#
# displacement increases
# AND
# load decreases
#
# because this combination is more interesting
# mechanically than an isolated load change.
# ============================================================

disp_score = (
    np.abs(
        analysis[
            "Smooth_Displacement_Change"
        ]
    )
    /
    disp_threshold
)

load_score = (
    np.abs(
        analysis[
            "Smooth_Load_Change"
        ]
    )
    /
    load_threshold
)


analysis[
    "Event_Score"
] = (
    disp_score +
    load_score
)


# Extra mechanical score
mechanical_condition = (
    (
        analysis[
            "Smooth_Displacement_Change"
        ] > 0
    )
    &
    (
        analysis[
            "Smooth_Load_Change"
        ] < 0
    )
)

analysis.loc[
    mechanical_condition,
    "Event_Score"
] *= 1.5


# ============================================================
# CANDIDATE EVENTS
# ============================================================

candidate_mask = (
    (
        np.abs(
            analysis[
                "Smooth_Displacement_Change"
            ]
        )
        >= disp_threshold
    )
    |
    (
        np.abs(
            analysis[
                "Smooth_Load_Change"
            ]
        )
        >= load_threshold
    )
)

candidates = analysis[
    candidate_mask
].copy()


# ============================================================
# SORT BY SCORE
# ============================================================

candidates = candidates.sort_values(
    "Event_Score",
    ascending=False
)


# ============================================================
# AVAILABLE IMAGE FRAMES
# ============================================================

def get_frame_number(filename):

    numbers = re.findall(
        r"\d+",
        os.path.basename(filename)
    )

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


image_files = glob.glob(
    os.path.join(
        IMAGE_DIR,
        "*.tif"
    )
)

available_frames = set()

for file in image_files:

    frame = get_frame_number(
        file
    )

    if frame is not None:

        available_frames.add(
            frame
        )


print(
    "\nAvailable image frames:",
    len(available_frames)
)


# ============================================================
# IMAGE VERIFICATION
# ============================================================

candidates[
    "Image_Available"
] = candidates[
    "Image Number"
].astype(int).isin(
    available_frames
)


# ============================================================
# DIC INTERVAL VERIFICATION
# ============================================================

dic_intervals = []

for file in glob.glob(
    os.path.join(
        STAGE6C_DIR,
        "dic_*.csv"
    )
):

    name = os.path.basename(
        file
    )

    match = re.search(
        r"dic_(\d+)_(\d+)\.csv",
        name
    )

    if match:

        dic_intervals.append(
            (
                int(match.group(1)),
                int(match.group(2))
            )
        )


def find_dic_interval(frame):

    for start, end in dic_intervals:

        if start <= frame <= end:

            return (
                f"{start}->{end}"
            )

    return ""


candidates[
    "DIC_Interval"
] = candidates[
    "Image Number"
].astype(int).apply(
    find_dic_interval
)


candidates[
    "DIC_Available"
] = (
    candidates[
        "DIC_Interval"
    ] != ""
)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_event(row):

    image_available = row[
        "Image_Available"
    ]

    dic_available = row[
        "DIC_Available"
    ]

    load_change = row[
        "Smooth_Load_Change"
    ]

    disp_change = row[
        "Smooth_Displacement_Change"
    ]

    if not image_available:

        return "Excel-only / image unavailable"

    if dic_available:

        if (
            load_change < 0
            and disp_change > 0
        ):

            return "DIC + mechanical candidate"

        return "Image + DIC candidate"

    return "Image-verifiable candidate"


candidates[
    "Event_Type"
] = candidates.apply(
    classify_event,
    axis=1
)


# ============================================================
# SAVE ALL CANDIDATES
# ============================================================

candidate_file = os.path.join(
    OUTPUT_DIR,
    "final_event_candidates.csv"
)

candidates.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# TOP EVENTS
# ============================================================

display_columns = [
    "Image Number",
    "Time_0",
    "Load ch 2 kN",
    PRIMARY_DISPLACEMENT,
    "Smooth_Load_Change",
    "Smooth_Displacement_Change",
    "Load_Rate",
    "Displacement_Rate",
    "Event_Score",
    "Image_Available",
    "DIC_Interval",
    "Event_Type"
]

print("\n==============================================")
print(" TOP EVENT CANDIDATES")
print("==============================================")

print(
    candidates[
        display_columns
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# AVAILABLE-IMAGE EVENTS
# ============================================================

image_events = candidates[
    candidates[
        "Image_Available"
    ]
].copy()

image_events_file = os.path.join(
    OUTPUT_DIR,
    "image_verifiable_events.csv"
)

image_events.to_csv(
    image_events_file,
    index=False
)


# ============================================================
# DIC-SUPPORTED EVENTS
# ============================================================

dic_events = candidates[
    candidates[
        "DIC_Available"
    ]
].copy()

dic_events_file = os.path.join(
    OUTPUT_DIR,
    "dic_supported_events.csv"
)

dic_events.to_csv(
    dic_events_file,
    index=False
)


# ============================================================
# PLOT 1
# LOAD VS TIME
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    analysis["Time_0"],
    analysis["Load_Analysis"]
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Load (kN)"
)

plt.title(
    "Load vs Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_load_vs_time.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# PLOT 2
# DISPLACEMENT VS TIME
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    analysis["Time_0"],
    analysis["Displacement_Analysis"]
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Displacement (µm)"
)

plt.title(
    "Displacement vs Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_displacement_vs_time.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# PLOT 3
# LOAD VS DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    analysis["Displacement_Analysis"],
    analysis["Load_Analysis"]
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

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_load_vs_displacement.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# PLOT 4
# EVENT SCORES
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    analysis["Image Number"],
    analysis["Event_Score"]
)

plt.xlabel(
    "Image Number"
)

plt.ylabel(
    "Event score"
)

plt.title(
    "Automatic Event Detection Score"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "event_score_vs_image.png"
    ),
    dpi=180
)

plt.close()


# ============================================================
# SAVE COMPLETE ANALYSIS
# ============================================================

analysis_file = os.path.join(
    OUTPUT_DIR,
    "complete_experimental_analysis.csv"
)

analysis.to_csv(
    analysis_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print(" FINAL SUMMARY")
print("==============================================")

print(
    "Total experimental rows:",
    len(analysis)
)

print(
    "Total event candidates:",
    len(candidates)
)

print(
    "Image-verifiable candidates:",
    len(image_events)
)

print(
    "DIC-supported candidates:",
    len(dic_events)
)

print(
    "\nSaved files:"
)

print(
    candidate_file
)

print(
    image_events_file
)

print(
    dic_events_file
)

print(
    analysis_file
)

print(
    "\n=============================================="
)

print(
    "STAGE 7 COMPLETE"
)

print(
    "=============================================="
)