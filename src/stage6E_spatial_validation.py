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

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

DIC_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage6C_Crack_Progression"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage6E_Spatial_Validation"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# STAGES TO VALIDATE
# ============================================================

STAGES = [
    (1274, 1500),
    (2000, 2500),
    (3000, 3432)
]


# ============================================================
# DIC / STRAIN SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

GRID_STEP = 10

GAUSSIAN_SIGMA = 2


# ============================================================
# LOADING / FIXTURE REGION
#
# IMPORTANT:
# We DO NOT remove this region from the heatmap.
# We only use it as a warning/exclusion zone when finding
# the maximum strain for interpretation.
# ============================================================

FIXTURE_X_MIN = 900
FIXTURE_X_MAX = 1080

FIXTURE_Y_MIN = 450
FIXTURE_Y_MAX = 638


# ============================================================
# CUSTOM DIC HEATMAP
#
# Purple -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red
# ============================================================

heatmap_colors = [
    (0.20, 0.00, 0.50),
    (0.00, 0.20, 0.90),
    (0.00, 0.80, 0.90),
    (0.20, 0.80, 0.20),
    (1.00, 1.00, 0.00),
    (1.00, 0.50, 0.00),
    (0.90, 0.00, 0.00)
]

dic_cmap = LinearSegmentedColormap.from_list(
    "dic_heatmap",
    heatmap_colors,
    N=256
)


# ============================================================
# FIND FRAME NUMBER FROM FILENAME
# Example:
# BGSL-2-1274_0.tif
#
# The actual frame number is the SECOND-LAST number.
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# BUILD IMAGE MAP
# ============================================================

def build_image_map():

    files = glob.glob(
        os.path.join(ROI_FOLDER, "*.tif")
    )

    frame_map = {}

    for file in files:

        frame_number = get_frame_number(file)

        if frame_number is None:
            continue

        if frame_number == 0:
            continue

        frame_map[frame_number] = file

    return frame_map


# ============================================================
# CALCULATE STRAIN FROM DIC CSV
# ============================================================

def calculate_strain(csv_path):

    df = pd.read_csv(csv_path)

    print("\nReading:")
    print(os.path.basename(csv_path))

    print("Original DIC points:", len(df))

    # --------------------------------------------------------
    # ZNCC filtering
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ].copy()

    print(
        "Points after ZNCC filtering:",
        len(df)
    )

    if len(df) < 100:

        print("WARNING: Too few DIC points.")

        return None


    # --------------------------------------------------------
    # Extract DIC coordinates and displacement
    # --------------------------------------------------------

    x = df["X"].values
    y = df["Y"].values

    u = df["dx"].values
    v = df["dy"].values


    # --------------------------------------------------------
    # Create regular grid
    # --------------------------------------------------------

    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    grid_x = np.arange(
        xmin,
        xmax + GRID_STEP,
        GRID_STEP
    )

    grid_y = np.arange(
        ymin,
        ymax + GRID_STEP,
        GRID_STEP
    )

    X, Y = np.meshgrid(
        grid_x,
        grid_y
    )


    # --------------------------------------------------------
    # Interpolate displacement
    # --------------------------------------------------------

    U = griddata(
        (x, y),
        u,
        (X, Y),
        method="linear"
    )

    V = griddata(
        (x, y),
        v,
        (X, Y),
        method="linear"
    )


    # --------------------------------------------------------
    # Fill interpolation holes using nearest neighbour
    # --------------------------------------------------------

    U_nearest = griddata(
        (x, y),
        u,
        (X, Y),
        method="nearest"
    )

    V_nearest = griddata(
        (x, y),
        v,
        (X, Y),
        method="nearest"
    )

    U = np.where(
        np.isnan(U),
        U_nearest,
        U
    )

    V = np.where(
        np.isnan(V),
        V_nearest,
        V
    )


    # --------------------------------------------------------
    # Smooth displacement field
    # --------------------------------------------------------

    U = gaussian_filter(
        U,
        sigma=GAUSSIAN_SIGMA
    )

    V = gaussian_filter(
        V,
        sigma=GAUSSIAN_SIGMA
    )


    # --------------------------------------------------------
    # Calculate displacement gradients
    # --------------------------------------------------------

    du_dy, du_dx = np.gradient(
        U,
        GRID_STEP,
        GRID_STEP
    )

    dv_dy, dv_dx = np.gradient(
        V,
        GRID_STEP,
        GRID_STEP
    )


    # --------------------------------------------------------
    # Strain tensor
    # --------------------------------------------------------

    epsilon_xx = du_dx

    epsilon_yy = dv_dy

    epsilon_xy = 0.5 * (
        du_dy + dv_dx
    )


    # --------------------------------------------------------
    # Principal strains
    # --------------------------------------------------------

    mean_strain = 0.5 * (
        epsilon_xx + epsilon_yy
    )

    radius = np.sqrt(
        (
            0.5 *
            (epsilon_xx - epsilon_yy)
        ) ** 2
        +
        epsilon_xy ** 2
    )

    principal_1 = mean_strain + radius

    principal_2 = mean_strain - radius


    # Absolute maximum principal strain
    principal_strain = np.maximum(
        np.abs(principal_1),
        np.abs(principal_2)
    )


    return X, Y, principal_strain


# ============================================================
# FIND MAXIMUM OUTSIDE LOADING ZONE
# ============================================================

def analyze_outside_fixture(X, Y, strain):

    fixture_mask = (
        (X >= FIXTURE_X_MIN) &
        (X <= FIXTURE_X_MAX) &
        (Y >= FIXTURE_Y_MIN) &
        (Y <= FIXTURE_Y_MAX)
    )

    valid = np.isfinite(strain)

    outside_mask = (
        valid &
        (~fixture_mask)
    )

    outside_values = strain[outside_mask]

    if len(outside_values) == 0:

        return None

    max_index = np.argmax(
        np.where(
            outside_mask,
            strain,
            -np.inf
        )
    )

    max_y_index, max_x_index = np.unravel_index(
        max_index,
        strain.shape
    )

    max_strain = strain[
        max_y_index,
        max_x_index
    ]

    max_x = X[
        max_y_index,
        max_x_index
    ]

    max_y = Y[
        max_y_index,
        max_x_index
    ]

    percentile_95 = np.percentile(
        outside_values,
        95
    )

    percentile_99 = np.percentile(
        outside_values,
        99
    )

    return {
        "max_strain": max_strain,
        "max_x": max_x,
        "max_y": max_y,
        "p95": percentile_95,
        "p99": percentile_99
    }


# ============================================================
# GENERATE ONE VALIDATION FIGURE
# ============================================================

def create_validation_figure(
    frame_start,
    frame_end,
    image_path,
    X,
    Y,
    strain
):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Could not read image:",
            image_path
        )

        return None


    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    stats = analyze_outside_fixture(
        X,
        Y,
        strain
    )

    if stats is None:

        print("Could not calculate statistics.")

        return None


    print("\nSpatial validation:")
    print(
        "Maximum strain OUTSIDE loading zone:",
        stats["max_strain"]
    )

    print(
        "Location:",
        stats["max_x"],
        stats["max_y"]
    )

    print(
        "95th percentile:",
        stats["p95"]
    )

    print(
        "99th percentile:",
        stats["p99"]
    )


    # --------------------------------------------------------
    # Use 99th percentile for visualization
    # --------------------------------------------------------

    finite_values = strain[
        np.isfinite(strain)
    ]

    vmax = np.percentile(
        finite_values,
        99
    )

    if vmax <= 0:

        vmax = np.max(
            finite_values
        )

    if vmax <= 0:

        vmax = 1e-6


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )


    # ========================================================
    # PANEL 1 — ORIGINAL IMAGE
    # ========================================================

    axes[0].imshow(
        image_rgb
    )

    axes[0].set_title(
        f"Original Concrete Image\nFrame {frame_end}"
    )

    axes[0].set_xlabel(
        "X coordinate (ROI pixels)"
    )

    axes[0].set_ylabel(
        "Y coordinate (ROI pixels)"
    )


    # Draw loading zone

    rectangle = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linewidth=2,
        linestyle="--"
    )

    axes[0].add_patch(
        rectangle
    )

    axes[0].text(
        FIXTURE_X_MIN,
        FIXTURE_Y_MIN - 8,
        "Loading / fixture zone",
        fontsize=9
    )


    # ========================================================
    # PANEL 2 — HEATMAP OVERLAY
    # ========================================================

    axes[1].imshow(
        image_rgb
    )

    heat = axes[1].pcolormesh(
        X,
        Y,
        strain,
        cmap=dic_cmap,
        shading="auto",
        alpha=0.65,
        vmin=0,
        vmax=vmax
    )

    axes[1].set_title(
        f"DIC Principal Strain Heatmap\n"
        f"{frame_start} → {frame_end}"
    )

    axes[1].set_xlabel(
        "X coordinate (ROI pixels)"
    )

    axes[1].set_ylabel(
        "Y coordinate (ROI pixels)"
    )

    plt.colorbar(
        heat,
        ax=axes[1],
        label="Absolute maximum principal strain"
    )


    # Draw loading zone

    rectangle2 = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linewidth=2,
        linestyle="--"
    )

    axes[1].add_patch(
        rectangle2
    )


    # Mark maximum OUTSIDE fixture

    axes[1].scatter(
        stats["max_x"],
        stats["max_y"],
        marker="x",
        s=80,
        linewidths=2
    )

    axes[1].text(
        stats["max_x"] + 20,
        stats["max_y"],
        "Max outside loading zone",
        fontsize=8
    )


    # ========================================================
    # PANEL 3 — CENTRAL REGION ZOOM
    # ========================================================

    axes[2].imshow(
        image_rgb
    )

    heat2 = axes[2].pcolormesh(
        X,
        Y,
        strain,
        cmap=dic_cmap,
        shading="auto",
        alpha=0.70,
        vmin=0,
        vmax=vmax
    )

    axes[2].set_title(
        "Central Deformation / Crack Region"
    )

    axes[2].set_xlabel(
        "X coordinate (ROI pixels)"
    )

    axes[2].set_ylabel(
        "Y coordinate (ROI pixels)"
    )


    # Zoom around expected crack region

    axes[2].set_xlim(
        650,
        1250
    )

    axes[2].set_ylim(
        600,
        200
    )


    # Draw loading zone

    rectangle3 = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linewidth=2,
        linestyle="--"
    )

    axes[2].add_patch(
        rectangle3
    )


    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"Stage 6E — Spatial DIC Validation\n"
        f"Frame {frame_start} → {frame_end}",
        fontsize=14
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"spatial_validation_{frame_start}_{frame_end}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nSaved:",
        output_file
    )


    return stats


# ============================================================
# MAIN
# ============================================================

print("==============================================")
print("      STAGE 6E — SPATIAL VALIDATION")
print("==============================================")


image_map = build_image_map()

print(
    "\nAvailable ROI frames:",
    len(image_map)
)


summary = []


for frame_start, frame_end in STAGES:

    print("\n==============================================")
    print(
        f"Validating {frame_start} -> {frame_end}"
    )
    print("==============================================")


    # --------------------------------------------------------
    # Check end image
    # --------------------------------------------------------

    if frame_end not in image_map:

        print(
            "End image unavailable:",
            frame_end
        )

        continue


    # --------------------------------------------------------
    # DIC CSV
    # --------------------------------------------------------

    csv_name = (
        f"dic_{frame_start:04d}_{frame_end:04d}.csv"
    )

    csv_path = os.path.join(
        DIC_FOLDER,
        csv_name
    )


    if not os.path.exists(csv_path):

        print(
            "DIC CSV not found:",
            csv_path
        )

        continue


    # --------------------------------------------------------
    # Calculate strain
    # --------------------------------------------------------

    result = calculate_strain(
        csv_path
    )

    if result is None:

        continue


    X, Y, strain = result


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    stats = create_validation_figure(
        frame_start,
        frame_end,
        image_map[frame_end],
        X,
        Y,
        strain
    )


    if stats is not None:

        summary.append([
            frame_start,
            frame_end,
            stats["max_strain"],
            stats["max_x"],
            stats["max_y"],
            stats["p95"],
            stats["p99"]
        ])


# ============================================================
# SAVE SUMMARY
# ============================================================

if len(summary) > 0:

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Frame_Start",
            "Frame_End",
            "Max_Strain_Outside_Loading_Zone",
            "Max_X",
            "Max_Y",
            "95th_Percentile",
            "99th_Percentile"
        ]
    )


    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "stage6E_spatial_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print("\n==============================================")
    print("       STAGE 6E COMPLETE")
    print("==============================================")

    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved summary:",
        summary_path
    )


else:

    print("\nNo validation results generated.")


print("\n==============================================")
print("       END OF STAGE 6E")
print("==============================================")