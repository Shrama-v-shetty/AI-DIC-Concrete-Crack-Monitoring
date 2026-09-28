import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset"

EXCEL_PATH = os.path.join(
    BASE_DIR,
    "experimental_data",
    "combined_image_load_data_0_to_6600.xlsx"
)

DIC_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage9_Full_Sequential_DIC"
)

DIC_SUMMARY_PATH = os.path.join(
    DIC_DIR,
    "full_dic_summary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage10_DIC_Excel_Integration"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_numeric(series):
    """
    Convert values to numeric and replace infinity with NaN.
    """
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan
    )


def save_plot(filename):
    """
    Save the current matplotlib figure.
    """
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# STEP 1 — LOAD EXPERIMENTAL EXCEL DATA
# ============================================================

print("\n" + "=" * 60)
print("STAGE 10 — DIC + EXCEL INTEGRATION")
print("=" * 60)

print("\nLoading experimental Excel data...")

excel_df = pd.read_excel(EXCEL_PATH)

print(f"Excel rows loaded: {len(excel_df)}")
print("Excel columns:")
print(list(excel_df.columns))


# ============================================================
# STEP 2 — CLEAN IMAGE NUMBER
# ============================================================

excel_df["Image Number"] = clean_numeric(
    excel_df["Image Number"]
)

excel_df = excel_df.dropna(
    subset=["Image Number"]
).copy()

excel_df["Image Number"] = (
    excel_df["Image Number"]
    .astype(int)
)


# ============================================================
# STEP 3 — CLEAN IMPORTANT EXPERIMENTAL COLUMNS
# ============================================================

numeric_columns = [
    "Time_0",
    "Load ch 2 kN",
    "Displacement ch 5 µm",
    "Displacement ch 6 µm",
    "Displacement ch 7 µm",
    "Displacement ch 8 µm",
    "Stress MPa"
]

for column in numeric_columns:

    if column in excel_df.columns:
        excel_df[column] = clean_numeric(
            excel_df[column]
        )


# ============================================================
# STEP 4 — REMOVE DUPLICATE IMAGE NUMBERS
# ============================================================

excel_df = (
    excel_df
    .sort_values("Image Number")
    .drop_duplicates(
        subset=["Image Number"],
        keep="first"
    )
    .reset_index(drop=True)
)

print(f"Clean experimental rows: {len(excel_df)}")


# ============================================================
# STEP 5 — LOAD DIC SUMMARY
# ============================================================

print("\nLoading Stage 9 DIC summary...")

dic_df = pd.read_csv(DIC_SUMMARY_PATH)

print(f"DIC rows loaded: {len(dic_df)}")
print("DIC columns:")
print(list(dic_df.columns))


# ============================================================
# STEP 6 — IDENTIFY FRAME COLUMNS
# ============================================================

# Stage 9 should contain Frame_1 and Frame_2.
# If your column names are slightly different, this
# section will show them clearly.

required_dic_columns = [
    "Frame_1",
    "Frame_2"
]

for column in required_dic_columns:

    if column not in dic_df.columns:
        raise ValueError(
            f"Missing DIC column: {column}\n"
            f"Available columns: {list(dic_df.columns)}"
        )


dic_df["Frame_1"] = clean_numeric(
    dic_df["Frame_1"]
)

dic_df["Frame_2"] = clean_numeric(
    dic_df["Frame_2"]
)

dic_df = dic_df.dropna(
    subset=["Frame_1", "Frame_2"]
).copy()

dic_df["Frame_1"] = dic_df["Frame_1"].astype(int)
dic_df["Frame_2"] = dic_df["Frame_2"].astype(int)


# ============================================================
# STEP 7 — PREPARE EXPERIMENTAL LOOKUP TABLES
# ============================================================

exp_lookup = excel_df.set_index("Image Number")


# ============================================================
# STEP 8 — BUILD SYNCHRONIZED MASTER DATASET
# ============================================================

print("\nSynchronizing DIC pairs with experimental data...")

master_rows = []

for _, dic_row in dic_df.iterrows():

    frame_1 = int(dic_row["Frame_1"])
    frame_2 = int(dic_row["Frame_2"])

    if frame_1 not in exp_lookup.index:
        continue

    if frame_2 not in exp_lookup.index:
        continue

    exp_1 = exp_lookup.loc[frame_1]
    exp_2 = exp_lookup.loc[frame_2]

    row = {
        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------
        "Frame_1": frame_1,
        "Frame_2": frame_2,

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------
        "Time_1": exp_1["Time_0"],
        "Time_2": exp_2["Time_0"],

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------
        "Load_1_kN": exp_1["Load ch 2 kN"],
        "Load_2_kN": exp_2["Load ch 2 kN"],

        "Load_Delta_kN": (
            exp_2["Load ch 2 kN"]
            - exp_1["Load ch 2 kN"]
        ),

        # ----------------------------------------------------
        # Displacement CH 5
        # ----------------------------------------------------
        "Displacement_ch5_1_um":
            exp_1["Displacement ch 5 µm"],

        "Displacement_ch5_2_um":
            exp_2["Displacement ch 5 µm"],

        # ----------------------------------------------------
        # Displacement CH 6
        # ----------------------------------------------------
        "Displacement_ch6_1_um":
            exp_1["Displacement ch 6 µm"],

        "Displacement_ch6_2_um":
            exp_2["Displacement ch 6 µm"],

        # ----------------------------------------------------
        # PRIMARY DISPLACEMENT — CH 7
        # ----------------------------------------------------
        "Displacement_ch7_1_um":
            exp_1["Displacement ch 7 µm"],

        "Displacement_ch7_2_um":
            exp_2["Displacement ch 7 µm"],

        "Displacement_ch7_Delta_um": (
            exp_2["Displacement ch 7 µm"]
            - exp_1["Displacement ch 7 µm"]
        ),

        # ----------------------------------------------------
        # Displacement CH 8
        # ----------------------------------------------------
        "Displacement_ch8_1_um":
            exp_1["Displacement ch 8 µm"],

        "Displacement_ch8_2_um":
            exp_2["Displacement ch 8 µm"],

        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------
        "Stress_1_MPa":
            exp_1["Stress MPa"],

        "Stress_2_MPa":
            exp_2["Stress MPa"],
    }

    # --------------------------------------------------------
    # Add all DIC information
    # --------------------------------------------------------

    for column in dic_df.columns:

        if column in ["Frame_1", "Frame_2"]:
            continue

        row[f"DIC_{column}"] = dic_row[column]

    master_rows.append(row)


# ============================================================
# STEP 9 — CREATE MASTER DATAFRAME
# ============================================================

master_df = pd.DataFrame(master_rows)

print(f"\nMatched DIC + Excel pairs: {len(master_df)}")


# ============================================================
# STEP 10 — CLEAN MASTER DATA
# ============================================================

for column in master_df.columns:

    if column not in ["Frame_1", "Frame_2"]:

        master_df[column] = clean_numeric(
            master_df[column]
        )


# ============================================================
# STEP 11 — CALCULATE TIME DELTA
# ============================================================

master_df["Time_Delta_sec"] = (
    master_df["Time_2"]
    - master_df["Time_1"]
)


# ============================================================
# STEP 12 — DIC QUALITY FLAG
# ============================================================

if "DIC_Average_ZNCC" in master_df.columns:

    master_df["DIC_Quality"] = np.where(
        master_df["DIC_Average_ZNCC"] >= 0.99,
        "Good",
        "Review"
    )

elif "DIC_ZNCC" in master_df.columns:

    master_df["DIC_Quality"] = np.where(
        master_df["DIC_ZNCC"] >= 0.99,
        "Good",
        "Review"
    )


# ============================================================
# STEP 13 — SAVE MASTER CSV
# ============================================================

master_path = os.path.join(
    OUTPUT_DIR,
    "master_dic_excel.csv"
)

master_df.to_csv(
    master_path,
    index=False
)

print(f"\nSaved master dataset:")
print(master_path)


# ============================================================
# STEP 14 — GRAPH 1
# LOAD VS DISPLACEMENT CH 7
# ============================================================

print("\nCreating Load vs Displacement graph...")

plt.figure(figsize=(10, 6))

plt.plot(
    excel_df["Displacement ch 7 µm"],
    excel_df["Load ch 2 kN"],
    linewidth=1.5
)

plt.xlabel("Displacement ch 7 (µm)")
plt.ylabel("Load (kN)")
plt.title("Load vs Displacement — Channel 7")

plt.grid(True, alpha=0.3)

save_plot(
    "load_vs_displacement_ch7.png"
)


# ============================================================
# STEP 15 — GRAPH 2
# TIME VS DISPLACEMENT CH 7
# ============================================================

print("Creating Time vs Displacement graph...")

plt.figure(figsize=(10, 6))

plt.plot(
    excel_df["Time_0"],
    excel_df["Displacement ch 7 µm"],
    linewidth=1.5
)

plt.xlabel("Time")
plt.ylabel("Displacement ch 7 (µm)")
plt.title("Time vs Displacement — Channel 7")

plt.grid(True, alpha=0.3)

save_plot(
    "time_vs_displacement_ch7.png"
)


# ============================================================
# STEP 16 — GRAPH 3
# TIME VS LOAD
# ============================================================

print("Creating Time vs Load graph...")

plt.figure(figsize=(10, 6))

plt.plot(
    excel_df["Time_0"],
    excel_df["Load ch 2 kN"],
    linewidth=1.5
)

plt.xlabel("Time")
plt.ylabel("Load (kN)")
plt.title("Time vs Load")

plt.grid(True, alpha=0.3)

save_plot(
    "time_vs_load.png"
)


# ============================================================
# STEP 17 — DIC INCREMENTAL DISPLACEMENT VS TIME
# ============================================================

if "DIC_Average_Magnitude" in master_df.columns:

    print("Creating DIC deformation vs time graph...")

    plt.figure(figsize=(10, 6))

    plt.plot(
        master_df["Time_2"],
        master_df["DIC_Average_Magnitude"],
        linewidth=1.0
    )

    plt.xlabel("Time")
    plt.ylabel(
        "DIC incremental displacement magnitude (px/pair)"
    )

    plt.title(
        "DIC Incremental Deformation vs Time"
    )

    plt.grid(True, alpha=0.3)

    save_plot(
        "dic_incremental_displacement_vs_time.png"
    )


# ============================================================
# STEP 18 — DIC INCREMENTAL DISPLACEMENT VS LOAD
# ============================================================

if "DIC_Average_Magnitude" in master_df.columns:

    print("Creating DIC deformation vs load graph...")

    plt.figure(figsize=(10, 6))

    plt.scatter(
        master_df["Load_2_kN"],
        master_df["DIC_Average_Magnitude"],
        s=5
    )

    plt.xlabel("Load (kN)")
    plt.ylabel(
        "DIC incremental displacement magnitude (px/pair)"
    )

    plt.title(
        "DIC Incremental Deformation vs Load"
    )

    plt.grid(True, alpha=0.3)

    save_plot(
        "dic_incremental_displacement_vs_load.png"
    )


# ============================================================
# STEP 19 — DIC QUALITY VS FRAME
# ============================================================

zncc_column = None

if "DIC_Average_ZNCC" in master_df.columns:
    zncc_column = "DIC_Average_ZNCC"

elif "DIC_ZNCC" in master_df.columns:
    zncc_column = "DIC_ZNCC"


if zncc_column is not None:

    print("Creating DIC quality graph...")

    plt.figure(figsize=(10, 6))

    plt.plot(
        master_df["Frame_2"],
        master_df[zncc_column],
        linewidth=1.0
    )

    plt.xlabel("Frame")
    plt.ylabel("Average ZNCC")
    plt.title("DIC Matching Quality vs Frame")

    plt.grid(True, alpha=0.3)

    save_plot(
        "dic_quality_vs_frame.png"
    )


# ============================================================
# STEP 20 — FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STAGE 10 COMPLETE")
print("=" * 60)

print(f"Experimental rows : {len(excel_df)}")
print(f"DIC pairs         : {len(dic_df)}")
print(f"Matched pairs     : {len(master_df)}")

print("\nPrimary displacement:")
print("Displacement ch 7 (µm)")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMaster dataset:")
print(master_path)

print("\nNext stage:")
print("DIC + experimental event correlation")