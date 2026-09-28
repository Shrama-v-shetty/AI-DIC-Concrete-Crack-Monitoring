import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 12D - COMBINED SIGNIFICANT-REGION PROGRESSION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

STAGE12A_DIR = os.path.join(
    BASE_DIR,
    "Stage12A_Significant_Region_Validation"
)

STAGE12B_DIR = os.path.join(
    BASE_DIR,
    "Stage12B_Significant_Region_Validation"
)

STAGE12C_DIR = os.path.join(
    BASE_DIR,
    "Stage12C_Significant_Region_Validation"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage12D_Combined_Progression"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INPUT SUMMARY FILES
# ============================================================

SUMMARY_FILES = [
    os.path.join(
        STAGE12A_DIR,
        "summary_1274_1500.csv"
    ),
    os.path.join(
        STAGE12B_DIR,
        "summary_2000_2500.csv"
    ),
    os.path.join(
        STAGE12C_DIR,
        "summary_3000_3432.csv"
    )
]


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("STAGE 12D - COMBINED SIGNIFICANT-REGION PROGRESSION")
print("=" * 70)

for path in SUMMARY_FILES:

    print()
    print("Checking:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Summary file not found:\n{path}"
        )


# ============================================================
# LOAD SUMMARY DATA
# ============================================================

dataframes = []

for path in SUMMARY_FILES:

    df = pd.read_csv(path)

    if df.empty:

        raise RuntimeError(
            f"Summary file is empty:\n{path}"
        )

    dataframes.append(df)


combined = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================================
# SORT BY FRAME
# ============================================================

combined = combined.sort_values(
    by=["Frame_1", "Frame_2"]
).reset_index(drop=True)


# ============================================================
# ADD STAGE LABEL
# ============================================================

combined["Stage"] = [
    "Early deformation",
    "Intermediate deformation",
    "Advanced deformation"
]


# ============================================================
# ADD CENTROID SHIFT
# ============================================================

first_x = combined.loc[
    0,
    "High_Strain_Centroid_X"
]

first_y = combined.loc[
    0,
    "High_Strain_Centroid_Y"
]

combined["Centroid_Shift_X_from_Start"] = (
    combined["High_Strain_Centroid_X"]
    - first_x
)

combined["Centroid_Shift_Y_from_Start"] = (
    combined["High_Strain_Centroid_Y"]
    - first_y
)


# ============================================================
# ADD STRAIN GROWTH RELATIVE TO FIRST STAGE
# ============================================================

first_mean = combined.loc[
    0,
    "Mean_Principal_Strain"
]

first_p99 = combined.loc[
    0,
    "P99_Principal_Strain"
]

first_max = combined.loc[
    0,
    "Max_Principal_Strain"
]

combined["Mean_Strain_Ratio"] = (
    combined["Mean_Principal_Strain"]
    / first_mean
)

combined["P99_Strain_Ratio"] = (
    combined["P99_Principal_Strain"]
    / first_p99
)

combined["Max_Strain_Ratio"] = (
    combined["Max_Principal_Strain"]
    / first_max
)


# ============================================================
# PRINT PROGRESSION TABLE
# ============================================================

print()
print("COMBINED PROGRESSION")
print()

display_columns = [
    "Stage",
    "Frame_1",
    "Frame_2",
    "Mean_Principal_Strain",
    "P95_Principal_Strain",
    "P99_Principal_Strain",
    "Max_Principal_Strain",
    "High_Strain_Centroid_X",
    "High_Strain_Centroid_Y",
    "Load_Delta_kN",
    "Displacement_ch7_Delta_um"
]

print(
    combined[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE COMBINED CSV
# ============================================================

combined_csv = os.path.join(
    OUTPUT_DIR,
    "stage12_combined_progression.csv"
)

combined.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# PRINT KEY OBSERVATIONS
# ============================================================

print()
print("KEY PROGRESSION")
print()

print(
    "Initial centroid X : "
    f"{first_x:.1f}"
)

print(
    "Final centroid X   : "
    f"{combined.loc[len(combined)-1, 'High_Strain_Centroid_X']:.1f}"
)

print(
    "Total centroid X shift : "
    f"{combined.loc[len(combined)-1, 'Centroid_Shift_X_from_Start']:.1f} pixels"
)

print(
    "Initial mean strain : "
    f"{first_mean:.6f}"
)

print(
    "Final mean strain   : "
    f"{combined.loc[len(combined)-1, 'Mean_Principal_Strain']:.6f}"
)

print(
    "Initial P99 strain  : "
    f"{first_p99:.6f}"
)

print(
    "Final P99 strain    : "
    f"{combined.loc[len(combined)-1, 'P99_Principal_Strain']:.6f}"
)

print(
    "Initial maximum strain : "
    f"{first_max:.6f}"
)

print(
    "Final maximum strain   : "
    f"{combined.loc[len(combined)-1, 'Max_Principal_Strain']:.6f}"
)


# ============================================================
# FIGURE 1
# MEAN STRAIN PROGRESSION
# ============================================================

stage_numbers = [1, 2, 3]

stage_labels = [
    "1274 → 1500",
    "2000 → 2500",
    "3000 → 3432"
]

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stage_numbers,
    combined["Mean_Principal_Strain"],
    marker="o",
    linewidth=2
)

plt.xticks(
    stage_numbers,
    stage_labels
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "Mean principal strain"
)

plt.title(
    "Progression of Mean DIC Principal Strain"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

mean_strain_plot = os.path.join(
    OUTPUT_DIR,
    "mean_strain_progression.png"
)

plt.savefig(
    mean_strain_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# P99 STRAIN PROGRESSION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stage_numbers,
    combined["P99_Principal_Strain"],
    marker="o",
    linewidth=2
)

plt.xticks(
    stage_numbers,
    stage_labels
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "P99 principal strain"
)

plt.title(
    "Progression of High-End DIC Strain"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

p99_plot = os.path.join(
    OUTPUT_DIR,
    "p99_strain_progression.png"
)

plt.savefig(
    p99_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# MAXIMUM STRAIN PROGRESSION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stage_numbers,
    combined["Max_Principal_Strain"],
    marker="o",
    linewidth=2
)

plt.xticks(
    stage_numbers,
    stage_labels
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "Maximum principal strain"
)

plt.title(
    "Progression of Maximum DIC Principal Strain"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

max_strain_plot = os.path.join(
    OUTPUT_DIR,
    "maximum_strain_progression.png"
)

plt.savefig(
    max_strain_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# CENTROID MOVEMENT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stage_numbers,
    combined["High_Strain_Centroid_X"],
    marker="o",
    linewidth=2
)

plt.xticks(
    stage_numbers,
    stage_labels
)

plt.xlabel(
    "Validation interval"
)

plt.ylabel(
    "High-strain centroid X (pixels)"
)

plt.title(
    "Movement of High-Strain Region"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

centroid_plot = os.path.join(
    OUTPUT_DIR,
    "high_strain_centroid_progression.png"
)

plt.savefig(
    centroid_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# DIC STRAIN + DISPLACEMENT
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    stage_numbers,
    combined["Mean_Principal_Strain"],
    marker="o",
    linewidth=2
)

ax1.set_xlabel(
    "Validation interval"
)

ax1.set_ylabel(
    "Mean principal strain"
)

ax1.set_xticks(
    stage_numbers
)

ax1.set_xticklabels(
    stage_labels
)

ax2 = ax1.twinx()

ax2.plot(
    stage_numbers,
    combined["Displacement_ch7_Delta_um"],
    marker="s",
    linewidth=2
)

ax2.set_ylabel(
    "Ch7 displacement change (µm)"
)

plt.title(
    "DIC Strain and Experimental Displacement Progression"
)

fig.tight_layout()

combined_response_plot = os.path.join(
    OUTPUT_DIR,
    "dic_strain_vs_displacement_progression.png"
)

plt.savefig(
    combined_response_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6
# DIC STRAIN + LOAD CHANGE
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    stage_numbers,
    combined["Mean_Principal_Strain"],
    marker="o",
    linewidth=2
)

ax1.set_xlabel(
    "Validation interval"
)

ax1.set_ylabel(
    "Mean principal strain"
)

ax1.set_xticks(
    stage_numbers
)

ax1.set_xticklabels(
    stage_labels
)

ax2 = ax1.twinx()

ax2.plot(
    stage_numbers,
    combined["Load_Delta_kN"],
    marker="s",
    linewidth=2
)

ax2.set_ylabel(
    "Load change (kN)"
)

plt.title(
    "DIC Strain and Experimental Load Change"
)

fig.tight_layout()

load_response_plot = os.path.join(
    OUTPUT_DIR,
    "dic_strain_vs_load_change.png"
)

plt.savefig(
    load_response_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY TEXT
# ============================================================

summary_text = os.path.join(
    OUTPUT_DIR,
    "stage12_progression_summary.txt"
)

with open(
    summary_text,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STAGE 12 COMBINED PROGRESSION SUMMARY\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        "Validation intervals:\n"
    )

    file.write(
        "1. 1274 -> 1500\n"
    )

    file.write(
        "2. 2000 -> 2500\n"
    )

    file.write(
        "3. 3000 -> 3432\n\n"
    )

    file.write(
        f"Initial high-strain centroid X: "
        f"{first_x:.1f}\n"
    )

    file.write(
        f"Final high-strain centroid X: "
        f"{combined.loc[len(combined)-1, 'High_Strain_Centroid_X']:.1f}\n"
    )

    file.write(
        f"Overall centroid X shift: "
        f"{combined.loc[len(combined)-1, 'Centroid_Shift_X_from_Start']:.1f} pixels\n\n"
    )

    file.write(
        f"Initial mean principal strain: "
        f"{first_mean:.6f}\n"
    )

    file.write(
        f"Final mean principal strain: "
        f"{combined.loc[len(combined)-1, 'Mean_Principal_Strain']:.6f}\n\n"
    )

    file.write(
        f"Initial P99 principal strain: "
        f"{first_p99:.6f}\n"
    )

    file.write(
        f"Final P99 principal strain: "
        f"{combined.loc[len(combined)-1, 'P99_Principal_Strain']:.6f}\n\n"
    )

    file.write(
        f"Initial maximum principal strain: "
        f"{first_max:.6f}\n"
    )

    file.write(
        f"Final maximum principal strain: "
        f"{combined.loc[len(combined)-1, 'Max_Principal_Strain']:.6f}\n\n"
    )

    file.write(
        "Interpretation:\n"
    )

    file.write(
        "The DIC results show progressive changes in "
        "localized deformation across the selected "
        "validation intervals. The centroid movement "
        "is reported as a spatial trend and should not "
        "be interpreted as standalone proof of crack "
        "propagation. Experimental load and displacement "
        "changes are retained for correlated mechanical "
        "interpretation.\n"
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("OUTPUTS")
print()

print(
    combined_csv
)

print(
    mean_strain_plot
)

print(
    p99_plot
)

print(
    max_strain_plot
)

print(
    centroid_plot
)

print(
    combined_response_plot
)

print(
    load_response_plot
)

print(
    summary_text
)

print()
print("=" * 70)
print("STAGE 12D COMPLETE")
print("=" * 70)