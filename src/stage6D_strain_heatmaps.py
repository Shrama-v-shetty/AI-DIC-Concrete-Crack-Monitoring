import cv2
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS
# ============================================================

ROI_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\BGSL_ROI"
)

DIC_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6C_Crack_Progression"
)

OUTPUT_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6D_Strain_Heatmaps"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# DIC intervals generated in Stage 6C
INTERVALS = [
    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432)
]


# Minimum correlation quality used for strain calculation.
# Stage 6C already used 0.98, but a slightly stricter
# threshold helps remove weak DIC points from the strain field.
MIN_ZNCC = 0.985


# Gaussian smoothing.
# This reduces small point-to-point DIC noise.
SMOOTH_SIGMA = 2.0


# Grid spacing used for interpolation.
GRID_STEP = 10


# ============================================================
# FIXTURE MASK
# ============================================================
#
# These coordinates are in the ROI coordinate system.
#
# The loading/fixture region was previously identified around
# this area.
#
# We exclude it from interpretation so that its movement does
# not dominate the crack/deformation heatmap.
#

FIXTURE_X_MIN = 930
FIXTURE_X_MAX = 1060

FIXTURE_Y_MIN = 490
FIXTURE_Y_MAX = 638


# ============================================================
# CREATE PURPLE -> BLUE -> CYAN -> GREEN -> YELLOW ->
# ORANGE -> RED COLORMAP
# ============================================================

heatmap_colors = [
    "#6A00A8",   # Purple
    "#2146D0",   # Blue
    "#00B7FF",   # Cyan
    "#00C853",   # Green
    "#F1E500",   # Yellow
    "#FF8C00",   # Orange
    "#D50000"    # Red
]

dic_cmap = LinearSegmentedColormap.from_list(
    "DIC_Deformation",
    heatmap_colors
)


# ============================================================
# GET ACTUAL FRAME NUMBER
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    # Example:
    # BGSL-2-1274_0.tif
    #
    # numbers = ['2', '1274', '0']
    #
    # Actual frame = second-last number

    return int(numbers[-2])


# ============================================================
# FIND ALL AVAILABLE ROI IMAGES
# ============================================================

all_image_files = glob.glob(
    os.path.join(ROI_FOLDER, "*.tif")
)

frame_files = {}

for file in all_image_files:

    frame = get_frame_number(file)

    if frame is None:
        continue

    if frame == 0:
        continue

    frame_files[frame] = file


print("==========================================")
print("   STAGE 6D - DIC STRAIN HEATMAPS")
print("==========================================")

print(
    "\nAvailable ROI images:",
    len(frame_files)
)


# ============================================================
# PROCESS EACH INTERVAL
# ============================================================

summary = []


for frame_start, frame_end in INTERVALS:

    print("\n==========================================")
    print(
        f"Processing: "
        f"{frame_start} -> {frame_end}"
    )
    print("==========================================")


    # --------------------------------------------------------
    # DIC CSV
    # --------------------------------------------------------

    dic_filename = (
        f"dic_{frame_start:04d}_{frame_end:04d}.csv"
    )

    dic_path = os.path.join(
        DIC_FOLDER,
        dic_filename
    )


    if not os.path.exists(dic_path):

        print(
            "DIC file not found:",
            dic_path
        )

        continue


    # --------------------------------------------------------
    # END IMAGE
    # --------------------------------------------------------

    if frame_end not in frame_files:

        print(
            "End image not available:",
            frame_end
        )

        continue


    # --------------------------------------------------------
    # READ DIC DATA
    # --------------------------------------------------------

    df = pd.read_csv(dic_path)


    print(
        "Original DIC points:",
        len(df)
    )


    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "X",
            "Y",
            "dx",
            "dy",
            "ZNCC"
        ]
    )


    # --------------------------------------------------------
    # ZNCC FILTER
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= MIN_ZNCC
    ].copy()


    print(
        "Points after ZNCC filtering:",
        len(df)
    )


    if len(df) < 100:

        print(
            "WARNING: Too few DIC points."
        )

        continue


    # --------------------------------------------------------
    # EXTRACT DIC ARRAYS
    # --------------------------------------------------------

    x = df["X"].to_numpy(
        dtype=float
    )

    y = df["Y"].to_numpy(
        dtype=float
    )

    u = df["dx"].to_numpy(
        dtype=float
    )

    v = df["dy"].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # CREATE REGULAR GRID
    # --------------------------------------------------------

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)


    grid_x, grid_y = np.meshgrid(
        np.arange(
            x_min,
            x_max + GRID_STEP,
            GRID_STEP
        ),
        np.arange(
            y_min,
            y_max + GRID_STEP,
            GRID_STEP
        )
    )


    # --------------------------------------------------------
    # INTERPOLATE U
    # --------------------------------------------------------

    grid_u = griddata(
        (x, y),
        u,
        (grid_x, grid_y),
        method="linear"
    )


    # --------------------------------------------------------
    # INTERPOLATE V
    # --------------------------------------------------------

    grid_v = griddata(
        (x, y),
        v,
        (grid_x, grid_y),
        method="linear"
    )


    # --------------------------------------------------------
    # FILL INTERPOLATION HOLES
    # --------------------------------------------------------

    valid_u = ~np.isnan(grid_u)

    if np.any(~valid_u):

        nearest_u = griddata(
            (x, y),
            u,
            (grid_x, grid_y),
            method="nearest"
        )

        grid_u[~valid_u] = (
            nearest_u[~valid_u]
        )


    valid_v = ~np.isnan(grid_v)

    if np.any(~valid_v):

        nearest_v = griddata(
            (x, y),
            v,
            (grid_x, grid_y),
            method="nearest"
        )

        grid_v[~valid_v] = (
            nearest_v[~valid_v]
        )


    # --------------------------------------------------------
    # SMOOTH DISPLACEMENT FIELD
    # --------------------------------------------------------

    grid_u = gaussian_filter(
        grid_u,
        sigma=SMOOTH_SIGMA
    )

    grid_v = gaussian_filter(
        grid_v,
        sigma=SMOOTH_SIGMA
    )


    # --------------------------------------------------------
    # CALCULATE SPATIAL GRADIENTS
    # --------------------------------------------------------
    #
    # u = horizontal displacement
    # v = vertical displacement
    #
    # du/dx, du/dy, dv/dx, dv/dy
    #
    # These derivatives are used to calculate strain.
    #

    du_dy, du_dx = np.gradient(
        grid_u,
        GRID_STEP,
        GRID_STEP
    )

    dv_dy, dv_dx = np.gradient(
        grid_v,
        GRID_STEP,
        GRID_STEP
    )


    # --------------------------------------------------------
    # STRAIN COMPONENTS
    # --------------------------------------------------------

    epsilon_xx = du_dx

    epsilon_yy = dv_dy

    epsilon_xy = (
        0.5 *
        (
            du_dy +
            dv_dx
        )
    )


    # --------------------------------------------------------
    # PRINCIPAL STRAIN
    # --------------------------------------------------------

    average_normal_strain = (
        0.5 *
        (
            epsilon_xx +
            epsilon_yy
        )
    )


    radius = np.sqrt(
        (
            0.5 *
            (
                epsilon_xx -
                epsilon_yy
            )
        ) ** 2
        +
        epsilon_xy ** 2
    )


    principal_strain_1 = (
        average_normal_strain +
        radius
    )


    principal_strain_2 = (
        average_normal_strain -
        radius
    )


    # --------------------------------------------------------
    # MAXIMUM ABSOLUTE PRINCIPAL STRAIN
    # --------------------------------------------------------

    principal_strain = np.maximum(
        np.abs(principal_strain_1),
        np.abs(principal_strain_2)
    )


    # --------------------------------------------------------
    # CREATE FIXTURE MASK
    # --------------------------------------------------------

    fixture_mask = (
        (grid_x >= FIXTURE_X_MIN) &
        (grid_x <= FIXTURE_X_MAX) &
        (grid_y >= FIXTURE_Y_MIN) &
        (grid_y <= FIXTURE_Y_MAX)
    )


    principal_strain_masked = (
        principal_strain.copy()
    )

    principal_strain_masked[
        fixture_mask
    ] = np.nan


    # --------------------------------------------------------
    # CALCULATE STATISTICS
    # --------------------------------------------------------

    valid_strain = (
        principal_strain_masked[
            np.isfinite(
                principal_strain_masked
            )
        ]
    )


    if len(valid_strain) == 0:

        print(
            "ERROR: No valid strain values."
        )

        continue


    max_strain = np.max(
        valid_strain
    )

    mean_strain = np.mean(
        valid_strain
    )

    percentile_95 = np.percentile(
        valid_strain,
        95
    )

    percentile_99 = np.percentile(
        valid_strain,
        99
    )


    # --------------------------------------------------------
    # LOCATION OF MAXIMUM STRAIN
    # --------------------------------------------------------

    max_index = np.nanargmax(
        principal_strain_masked
    )

    max_y_index, max_x_index = (
        np.unravel_index(
            max_index,
            principal_strain_masked.shape
        )
    )


    max_x = grid_x[
        max_y_index,
        max_x_index
    ]

    max_y = grid_y[
        max_y_index,
        max_x_index
    ]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nSTRAIN RESULT")
    print("------------------------------------------")

    print(
        "Maximum principal strain:",
        max_strain
    )

    print(
        "Mean principal strain:",
        mean_strain
    )

    print(
        "95th percentile:",
        percentile_95
    )

    print(
        "99th percentile:",
        percentile_99
    )

    print(
        "Maximum strain location:"
    )

    print(
        "    X =",
        round(max_x, 2)
    )

    print(
        "    Y =",
        round(max_y, 2)
    )


    # ========================================================
    # SAVE STRAIN CSV
    # ========================================================

    output_rows = []


    for row in range(
        grid_x.shape[0]
    ):

        for col in range(
            grid_x.shape[1]
        ):

            if np.isnan(
                principal_strain_masked[
                    row,
                    col
                ]
            ):
                continue


            output_rows.append([
                grid_x[row, col],
                grid_y[row, col],
                grid_u[row, col],
                grid_v[row, col],
                epsilon_xx[row, col],
                epsilon_yy[row, col],
                epsilon_xy[row, col],
                principal_strain_1[row, col],
                principal_strain_2[row, col],
                principal_strain_masked[
                    row,
                    col
                ]
            ])


    strain_df = pd.DataFrame(
        output_rows,
        columns=[
            "X",
            "Y",
            "U",
            "V",
            "Epsilon_xx",
            "Epsilon_yy",
            "Epsilon_xy",
            "Principal_Strain_1",
            "Principal_Strain_2",
            "Max_Absolute_Principal_Strain"
        ]
    )


    strain_csv = os.path.join(
        OUTPUT_FOLDER,
        f"strain_{frame_start:04d}_{frame_end:04d}.csv"
    )


    strain_df.to_csv(
        strain_csv,
        index=False
    )


    print(
        "\nSaved strain CSV:",
        strain_csv
    )


    # ========================================================
    # READ END IMAGE
    # ========================================================

    image_path = frame_files[
        frame_end
    ]


    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        print(
            "Could not read image:",
            image_path
        )

        continue


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    ax.imshow(
        image,
        cmap="gray",
        origin="upper"
    )


    # --------------------------------------------------------
    # Use percentile-based upper limit.
    #
    # This prevents one extreme noisy point from making
    # the rest of the heatmap appear almost uniform.
    # --------------------------------------------------------

    color_max = percentile_99


    if color_max <= 0:

        color_max = max_strain


    heatmap = ax.contourf(
        grid_x,
        grid_y,
        principal_strain_masked,
        levels=100,
        cmap=dic_cmap,
        vmin=0,
        vmax=color_max,
        alpha=0.65
    )


    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    colorbar = plt.colorbar(
        heatmap,
        ax=ax
    )


    colorbar.set_label(
        "Maximum absolute principal strain"
    )


    # --------------------------------------------------------
    # Mark maximum strain point
    # --------------------------------------------------------

    ax.scatter(
        max_x,
        max_y,
        marker="x",
        s=100,
        linewidths=2,
        label="Maximum strain"
    )


    ax.legend(
        loc="upper right"
    )


    ax.set_title(
        f"DIC Principal Strain Heatmap\n"
        f"Frame {frame_start} → {frame_end}"
    )


    ax.set_xlabel(
        "X coordinate (ROI pixels)"
    )

    ax.set_ylabel(
        "Y coordinate (ROI pixels)"
    )


    plt.tight_layout()


    # ========================================================
    # SAVE HEATMAP
    # ========================================================

    heatmap_path = os.path.join(
        OUTPUT_FOLDER,
        f"strain_heatmap_{frame_start:04d}_{frame_end:04d}.png"
    )


    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved heatmap:",
        heatmap_path
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary.append([
        frame_start,
        frame_end,
        len(df),
        max_strain,
        mean_strain,
        percentile_95,
        percentile_99,
        max_x,
        max_y
    ])


# ============================================================
# SAVE OVERALL SUMMARY
# ============================================================

if len(summary) > 0:

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Frame_Start",
            "Frame_End",
            "DIC_Points_Used",
            "Maximum_Principal_Strain",
            "Mean_Principal_Strain",
            "Strain_95th_Percentile",
            "Strain_99th_Percentile",
            "Maximum_Strain_X",
            "Maximum_Strain_Y"
        ]
    )


    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "strain_progression_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print("\n==========================================")
    print("       STAGE 6D COMPLETE")
    print("==========================================")


    print(
        "\nSummary saved:"
    )

    print(
        summary_path
    )


    print("\nSUMMARY")
    print("------------------------------------------")

    print(
        summary_df.to_string(
            index=False
        )
    )


else:

    print("\n==========================================")
    print("      NO STRAIN RESULTS GENERATED")
    print("==========================================")