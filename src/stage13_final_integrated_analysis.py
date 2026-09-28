import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 13 - FINAL INTEGRATED ANALYSIS ENGINE
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

STAGE10_DIR = os.path.join(
    BASE_DIR,
    "Stage10_DIC_Excel_Integration"
)

STAGE12_DIR = os.path.join(
    BASE_DIR,
    "Stage12D_Combined_Progression"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage13_Final_Integrated_Analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MASTER_CSV = os.path.join(
    STAGE10_DIR,
    "master_dic_excel.csv"
)

STAGE12_CSV = os.path.join(
    STAGE12_DIR,
    "stage12_combined_progression.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def choose_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STAGE 13 - FINAL INTEGRATED ANALYSIS ENGINE")
print("=" * 75)


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("CHECKING INPUT FILES")

if not os.path.exists(MASTER_CSV):

    raise FileNotFoundError(
        f"Stage 10 master dataset not found:\n{MASTER_CSV}"
    )

if not os.path.exists(STAGE12_CSV):

    raise FileNotFoundError(
        f"Stage 12 combined progression not found:\n{STAGE12_CSV}"
    )

print(
    "Stage 10 master dataset : FOUND"
)

print(
    "Stage 12 progression    : FOUND"
)


# ============================================================
# LOAD STAGE 10
# ============================================================

print()
print("LOADING STAGE 10 MASTER DATASET")

master = pd.read_csv(
    MASTER_CSV
)

print(
    f"Rows loaded: {len(master)}"
)

print(
    f"Columns loaded: {len(master.columns)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Frame_1",
    "Frame_2",
    "Time_1",
    "Time_2",
    "Load_1_kN",
    "Load_2_kN",
    "Load_Delta_kN",
    "DIC_Average_ZNCC",
    "DIC_Average_dx",
    "DIC_Average_dy",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude"
]

missing_required = [
    column
    for column in required_columns
    if column not in master.columns
]

if missing_required:

    print()
    print(
        "WARNING - Missing Stage 10 columns:"
    )

    for column in missing_required:
        print(
            "  ",
            column
        )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Frame_1",
    "Frame_2",
    "Time_1",
    "Time_2",
    "Load_1_kN",
    "Load_2_kN",
    "Load_Delta_kN",
    "DIC_Average_ZNCC",
    "DIC_Average_dx",
    "DIC_Average_dy",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude"
]

for column in numeric_columns:

    if column in master.columns:

        master[column] = safe_numeric(
            master[column]
        )


# ============================================================
# PRIMARY DISPLACEMENT
# ============================================================

disp_col_1 = choose_column(
    master,
    [
        "Displacement_ch7_1_um",
        "Displacement_ch7_1_µm",
        "Displacement_ch7_1"
    ]
)

disp_col_2 = choose_column(
    master,
    [
        "Displacement_ch7_2_um",
        "Displacement_ch7_2_µm",
        "Displacement_ch7_2"
    ]
)

disp_delta_col = choose_column(
    master,
    [
        "Displacement_ch7_Delta_um",
        "Displacement_ch7_Delta_µm",
        "Displacement_ch7_Delta"
    ]
)

print()
print("PRIMARY EXPERIMENTAL DISPLACEMENT")

print(
    "Start column:",
    disp_col_1
)

print(
    "End column  :",
    disp_col_2
)

print(
    "Delta column:",
    disp_delta_col
)

if disp_delta_col is not None:

    master["Primary_Displacement_Delta_um"] = safe_numeric(
        master[disp_delta_col]
    )

else:

    master["Primary_Displacement_Delta_um"] = np.nan


if disp_col_1 is not None:

    master["Primary_Displacement_1_um"] = safe_numeric(
        master[disp_col_1]
    )

else:

    master["Primary_Displacement_1_um"] = np.nan


if disp_col_2 is not None:

    master["Primary_Displacement_2_um"] = safe_numeric(
        master[disp_col_2]
    )

else:

    master["Primary_Displacement_2_um"] = np.nan


# ============================================================
# SORT DATA
# ============================================================

master = master.sort_values(
    "Frame_1"
).reset_index(
    drop=True
)


# ============================================================
# CLEAN INVALID VALUES
# ============================================================

analysis_columns = [
    "Load_Delta_kN",
    "Primary_Displacement_Delta_um",
    "DIC_Average_Magnitude",
    "DIC_Maximum_Magnitude",
    "DIC_Average_ZNCC"
]

for column in analysis_columns:

    if column in master.columns:

        master[column] = (
            master[column]
            .replace([np.inf, -np.inf], np.nan)
        )


# ============================================================
# DIC QUALITY
# ============================================================

if "DIC_Average_ZNCC" in master.columns:

    master["DIC_Quality_Category"] = pd.cut(
        master["DIC_Average_ZNCC"],
        bins=[
            -np.inf,
            0.98,
            0.985,
            0.995,
            np.inf
        ],
        labels=[
            "Low",
            "Acceptable",
            "Good",
            "Very good"
        ]
    )

else:

    master["DIC_Quality_Category"] = "Unknown"


# ============================================================
# EXPERIMENTAL CHANGE INDICATORS
# ============================================================

master["Load_Decrease"] = (
    master["Load_Delta_kN"] < 0
)

master["Displacement_Increase"] = (
    master["Primary_Displacement_Delta_um"] > 0
)

master["Load_Drop_and_Displacement_Rise"] = (
    master["Load_Decrease"]
    & master["Displacement_Increase"]
)


# ============================================================
# ROBUST THRESHOLDS
# ============================================================

print()
print("CALCULATING ROBUST CHANGE THRESHOLDS")

valid_load = master[
    "Load_Delta_kN"
].dropna()

valid_disp = master[
    "Primary_Displacement_Delta_um"
].dropna()

valid_dic_avg = master[
    "DIC_Average_Magnitude"
].dropna()

valid_dic_max = master[
    "DIC_Maximum_Magnitude"
].dropna()


if len(valid_load) > 0:

    load_abs_threshold = float(
        np.percentile(
            np.abs(valid_load),
            95
        )
    )

else:

    load_abs_threshold = np.nan


if len(valid_disp) > 0:

    disp_abs_threshold = float(
        np.percentile(
            np.abs(valid_disp),
            95
        )
    )

else:

    disp_abs_threshold = np.nan


if len(valid_dic_avg) > 0:

    dic_avg_threshold = float(
        np.percentile(
            valid_dic_avg,
            95
        )
    )

else:

    dic_avg_threshold = np.nan


if len(valid_dic_max) > 0:

    dic_max_threshold = float(
        np.percentile(
            valid_dic_max,
            95
        )
    )

else:

    dic_max_threshold = np.nan


print(
    f"95th percentile |ΔLoad|      : "
    f"{load_abs_threshold:.6f}"
)

print(
    f"95th percentile |ΔCh7|       : "
    f"{disp_abs_threshold:.6f}"
)

print(
    f"95th percentile DIC average  : "
    f"{dic_avg_threshold:.6f}"
)

print(
    f"95th percentile DIC maximum  : "
    f"{dic_max_threshold:.6f}"
)


# ============================================================
# SIGNIFICANT EXPERIMENTAL CHANGES
# ============================================================

master["Significant_Load_Change"] = (
    np.abs(master["Load_Delta_kN"])
    >= load_abs_threshold
)

master["Significant_Displacement_Change"] = (
    np.abs(
        master[
            "Primary_Displacement_Delta_um"
        ]
    )
    >= disp_abs_threshold
)


# ============================================================
# SIGNIFICANT DIC DEFORMATION
# ============================================================

master["Significant_DIC_Average"] = (
    master[
        "DIC_Average_Magnitude"
    ]
    >= dic_avg_threshold
)

master["Significant_DIC_Maximum"] = (
    master[
        "DIC_Maximum_Magnitude"
    ]
    >= dic_max_threshold
)


# ============================================================
# CORRELATED MECHANICAL CONDITIONS
# ============================================================

master["Correlated_Mechanical_Response"] = (
    master["Significant_Load_Change"]
    & master["Significant_Displacement_Change"]
)

master["Load_Drop_Displacement_Rise"] = (
    master["Load_Delta_kN"] < 0
) & (
    master[
        "Primary_Displacement_Delta_um"
    ] > 0
)


# ============================================================
# DIC + EXPERIMENTAL CORRELATION
# ============================================================

master["Experimental_DIC_Correlation"] = (
    master[
        "Correlated_Mechanical_Response"
    ]
    & (
        master[
            "Significant_DIC_Average"
        ]
        | master[
            "Significant_DIC_Maximum"
        ]
    )
)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_row(row):

    if bool(
        row["Experimental_DIC_Correlation"]
    ):

        return (
            "Strong correlated deformation"
        )

    if bool(
        row["Load_Drop_Displacement_Rise"]
    ) and (
        bool(
            row["Significant_DIC_Average"]
        )
        or bool(
            row["Significant_DIC_Maximum"]
        )
    ):

        return (
            "Load-drop/displacement-rise with DIC deformation"
        )

    if bool(
        row["Correlated_Mechanical_Response"]
    ):

        return (
            "Mechanical transition"
        )

    if (
        bool(
            row["Significant_DIC_Average"]
        )
        or bool(
            row["Significant_DIC_Maximum"]
        )
    ):

        return (
            "DIC deformation candidate"
        )

    return "Normal / lower-change interval"


master[
    "Final_Event_Category"
] = master.apply(
    classify_row,
    axis=1
)


# ============================================================
# EVENT PRIORITY
# ============================================================

master["Event_Priority"] = 0

master.loc[
    master[
        "Final_Event_Category"
    ] == "DIC deformation candidate",
    "Event_Priority"
] = 1

master.loc[
    master[
        "Final_Event_Category"
    ] == "Mechanical transition",
    "Event_Priority"
] = 2

master.loc[
    master[
        "Final_Event_Category"
    ] == (
        "Load-drop/displacement-rise "
        "with DIC deformation"
    ),
    "Event_Priority"
] = 3

master.loc[
    master[
        "Final_Event_Category"
    ] == "Strong correlated deformation",
    "Event_Priority"
] = 4


# ============================================================
# SELECT IMPORTANT EVENTS
# ============================================================

important_events = master[
    master["Event_Priority"] >= 2
].copy()

important_events = important_events.sort_values(
    [
        "Event_Priority",
        "Frame_1"
    ],
    ascending=[
        False,
        True
    ]
)


# ============================================================
# LOAD/DISPLACEMENT/DIC SUMMARY
# ============================================================

print()
print("FINAL INTEGRATED SUMMARY")

print(
    f"Total DIC intervals analysed : "
    f"{len(master)}"
)

print(
    f"Load-drop + displacement-rise intervals : "
    f"{int(master['Load_Drop_Displacement_Rise'].sum())}"
)

print(
    f"Mechanical transition intervals : "
    f"{int(master['Correlated_Mechanical_Response'].sum())}"
)

print(
    f"DIC deformation candidates : "
    f"{int((
        master['Significant_DIC_Average']
        | master['Significant_DIC_Maximum']
    ).sum())}"
)

print(
    f"Strong correlated deformation intervals : "
    f"{int(master['Experimental_DIC_Correlation'].sum())}"
)


# ============================================================
# SAVE FULL DATASET
# ============================================================

full_output = os.path.join(
    OUTPUT_DIR,
    "final_integrated_analysis.csv"
)

master.to_csv(
    full_output,
    index=False
)


# ============================================================
# SAVE IMPORTANT EVENTS
# ============================================================

events_output = os.path.join(
    OUTPUT_DIR,
    "important_deformation_events.csv"
)

important_events.to_csv(
    events_output,
    index=False
)


# ============================================================
# LOAD STAGE 12 PROGRESSION
# ============================================================

stage12 = pd.read_csv(
    STAGE12_CSV
)

stage12 = stage12.sort_values(
    "Frame_1"
).reset_index(
    drop=True
)


# ============================================================
# STAGE 12 PROGRESSION SUMMARY
# ============================================================

stage12_summary = {
    "Initial centroid X":
        stage12.loc[
            0,
            "High_Strain_Centroid_X"
        ],

    "Final centroid X":
        stage12.loc[
            len(stage12) - 1,
            "High_Strain_Centroid_X"
        ],

    "Centroid X shift":
        (
            stage12.loc[
                len(stage12) - 1,
                "High_Strain_Centroid_X"
            ]
            -
            stage12.loc[
                0,
                "High_Strain_Centroid_X"
            ]
        ),

    "Initial mean strain":
        stage12.loc[
            0,
            "Mean_Principal_Strain"
        ],

    "Final mean strain":
        stage12.loc[
            len(stage12) - 1,
            "Mean_Principal_Strain"
        ],

    "Initial P99 strain":
        stage12.loc[
            0,
            "P99_Principal_Strain"
        ],

    "Final P99 strain":
        stage12.loc[
            len(stage12) - 1,
            "P99_Principal_Strain"
        ],

    "Initial maximum strain":
        stage12.loc[
            0,
            "Max_Principal_Strain"
        ],

    "Final maximum strain":
        stage12.loc[
            len(stage12) - 1,
            "Max_Principal_Strain"
        ]
}


# ============================================================
# PLOT 1
# LOAD VS TIME
# ============================================================

if (
    "Time_2" in master.columns
    and "Load_2_kN" in master.columns
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        master["Time_2"],
        master["Load_2_kN"],
        linewidth=1.5
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Load (kN)"
    )

    plt.title(
        "Experimental Load vs Time"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "final_load_vs_time.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PLOT 2
# DISPLACEMENT VS TIME
# ============================================================

if (
    "Time_2" in master.columns
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        master["Time_2"],
        master[
            "Primary_Displacement_2_um"
        ],
        linewidth=1.5
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Displacement ch 7 (µm)"
    )

    plt.title(
        "Primary Experimental Displacement vs Time"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "final_displacement_vs_time.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PLOT 3
# LOAD VS DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    master[
        "Primary_Displacement_2_um"
    ],
    master[
        "Load_2_kN"
    ],
    linewidth=1.5
)

plt.xlabel(
    "Displacement ch 7 (µm)"
)

plt.ylabel(
    "Load (kN)"
)

plt.title(
    "Load vs Primary Displacement"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_load_vs_displacement.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 4
# DIC DEFORMATION VS FRAME
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    master[
        "Frame_2"
    ],
    master[
        "DIC_Average_Magnitude"
    ],
    linewidth=1.2
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Average DIC deformation (pixels)"
)

plt.title(
    "Incremental DIC Deformation vs Frame"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_dic_deformation_vs_frame.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 5
# DIC QUALITY
# ============================================================

if "DIC_Average_ZNCC" in master.columns:

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        master["Frame_2"],
        master["DIC_Average_ZNCC"],
        linewidth=1.2
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Average ZNCC"
    )

    plt.title(
        "DIC Tracking Quality"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "final_dic_quality_vs_frame.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PLOT 6
# STAGE 12 MEAN STRAIN PROGRESSION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(1, len(stage12) + 1),
    stage12[
        "Mean_Principal_Strain"
    ],
    marker="o",
    linewidth=2
)

plt.xticks(
    range(1, len(stage12) + 1),
    [
        f"{a} → {b}"
        for a, b
        in zip(
            stage12["Frame_1"],
            stage12["Frame_2"]
        )
    ]
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "Mean principal strain"
)

plt.title(
    "Validated DIC Strain Progression"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validated_strain_progression.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 7
# HIGH-STRAIN CENTROID MOVEMENT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(1, len(stage12) + 1),
    stage12[
        "High_Strain_Centroid_X"
    ],
    marker="o",
    linewidth=2
)

plt.xticks(
    range(1, len(stage12) + 1),
    [
        f"{a} → {b}"
        for a, b
        in zip(
            stage12["Frame_1"],
            stage12["Frame_2"]
        )
    ]
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "High-strain centroid X (pixels)"
)

plt.title(
    "Spatial Movement of High-Strain Region"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validated_centroid_progression.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GENERATE FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "final_integrated_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "FINAL INTEGRATED DIC ANALYSIS REPORT\n"
    )

    report.write(
        "===================================\n\n"
    )

    report.write(
        "DATASET\n"
    )

    report.write(
        f"Total DIC intervals analysed: "
        f"{len(master)}\n"
    )

    report.write(
        f"Stage 12 validation intervals: "
        f"{len(stage12)}\n\n"
    )

    report.write(
        "PRIMARY EXPERIMENTAL MEASUREMENT\n"
    )

    report.write(
        "Displacement channel: "
        "Displacement ch 7 (µm)\n\n"
    )

    report.write(
        "DIC QUALITY\n"
    )

    report.write(
        f"95th percentile average DIC magnitude: "
        f"{dic_avg_threshold:.6f}\n"
    )

    report.write(
        f"95th percentile maximum DIC magnitude: "
        f"{dic_max_threshold:.6f}\n\n"
    )

    report.write(
        "CORRELATED RESPONSE\n"
    )

    report.write(
        f"Load-drop + displacement-rise intervals: "
        f"{int(master['Load_Drop_Displacement_Rise'].sum())}\n"
    )

    report.write(
        f"Strong correlated deformation intervals: "
        f"{int(master['Experimental_DIC_Correlation'].sum())}\n\n"
    )

    report.write(
        "VALIDATED STAGE 12 PROGRESSION\n"
    )

    for key, value in stage12_summary.items():

        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\nINTERPRETATION\n"
    )

    report.write(
        "The integrated analysis combines experimental "
        "load/displacement response with incremental DIC "
        "deformation measurements. The selected Stage 12 "
        "validation intervals demonstrate progressive "
        "localized deformation and a leftward movement of "
        "the identified high-strain region. These results "
        "support the use of DIC as a spatial deformation "
        "monitoring component. A high-strain region is not "
        "automatically labelled as a crack without visual "
        "or additional crack-specific validation.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 75)
print("STAGE 13 COMPLETE")
print("=" * 75)

print()
print("FULL INTEGRATED DATASET:")
print(full_output)

print()
print("IMPORTANT EVENTS:")
print(events_output)

print()
print("FINAL REPORT:")
print(report_path)

print()
print("GENERATED PLOTS:")
print(
    os.path.join(
        OUTPUT_DIR,
        "final_load_vs_time.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_displacement_vs_time.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_load_vs_displacement.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_dic_deformation_vs_frame.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_dic_quality_vs_frame.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "validated_strain_progression.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "validated_centroid_progression.png"
    )
)

print()
print(
    "Next step: review the Stage 13 integrated results "
    "before building the final user-facing software interface."
)