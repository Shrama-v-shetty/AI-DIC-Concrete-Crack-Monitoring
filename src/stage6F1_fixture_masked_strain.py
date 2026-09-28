import os
import glob
import re
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\shrama\Desktop\Dam updated dataset"

DIC_DIR = os.path.join(
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
    "Stage6F1_Fixture_Masked"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

GRID_STEP = 10

GAUSSIAN_SIGMA = 2

# Stages already tested
INTERVALS = [
    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432),
]


# ============================================================
# FIXTURE POLYGON
#
# Based on the actual uploaded ROI image.
#
# Only the black loading fixture is masked.
# Concrete immediately above it remains available.
# ============================================================

FIXTURE_POLYGON = np.array([
    [962, 638],
    [955, 638],
    [958, 560],
    [960, 540],
    [963, 523],
    [968, 512],
    [978, 506],
    [990, 507],
    [999, 513],
    [1004, 525],
    [1007, 545],
    [1010, 638]
], dtype=np.int32)


# ============================================================
# COLOR MAP
#
# Purple -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red
# ============================================================

HEATMAP_COLORS = [
    "purple",
    "blue",
    "cyan",
    "green",
    "yellow",
    "orange",
    "red"
]

DIC_CMAP = LinearSegmentedColormap.from_list(
    "dic_strain",
    HEATMAP_COLORS,
    N=256
)


# ============================================================
# FIND IMAGE FRAME NUMBER
# ============================================================

def get_frame_number(filename):

    name = os.path.basename(filename)

    numbers = re.findall(r"\d+", name)

    if len(numbers) < 2:
        return None

    # Important:
    # actual frame number is the second-last number
    return int(numbers[-2])


# ============================================================
# LOAD AVAILABLE ROI IMAGES
# ============================================================

def load_image_map():

    files = glob.glob(
        os.path.join(IMAGE_DIR, "*.tif")
    )

    image_map = {}

    for file in files:

        frame = get_frame_number(file)

        if frame is not None:
            image_map[frame] = file

    return image_map


# ============================================================
# CHECK WHETHER POINT IS INSIDE FIXTURE
# ============================================================

def create_fixture_mask(height, width):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [FIXTURE_POLYGON],
        1
    )

    return mask


# ============================================================
# REMOVE FIXTURE DIC POINTS
# ============================================================

def remove_fixture_points(df, height, width):

    points = df[
        ["X", "Y"]
    ].values.astype(np.float32)

    fixture_mask = create_fixture_mask(
        height,
        width
    )

    x = points[:, 0].astype(int)
    y = points[:, 1].astype(int)

    valid_bounds = (
        (x >= 0) &
        (x < width) &
        (y >= 0) &
        (y < height)
    )

    inside_fixture = np.zeros(
        len(points),
        dtype=bool
    )

    inside_fixture[valid_bounds] = (
        fixture_mask[
            y[valid_bounds],
            x[valid_bounds]
        ] == 1
    )

    cleaned = df.loc[
        ~inside_fixture
    ].copy()

    return cleaned


# ============================================================
# CALCULATE STRAIN FIELD
# ============================================================

def calculate_strain(df, height, width):

    # --------------------------------------------------------
    # Create regular grid
    # --------------------------------------------------------

    x_values = np.arange(
        0,
        width,
        GRID_STEP
    )

    y_values = np.arange(
        0,
        height,
        GRID_STEP
    )

    grid_x, grid_y = np.meshgrid(
        x_values,
        y_values
    )

    points = df[
        ["X", "Y"]
    ].values

    u_values = df["dx"].values
    v_values = df["dy"].values

    # --------------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------------

    grid_u = griddata(
        points,
        u_values,
        (grid_x, grid_y),
        method="linear"
    )

    grid_v = griddata(
        points,
        v_values,
        (grid_x, grid_y),
        method="linear"
    )

    # --------------------------------------------------------
    # Fill interpolation holes
    # --------------------------------------------------------

    missing_u = np.isnan(grid_u)

    if np.any(missing_u):

        grid_u[missing_u] = griddata(
            points,
            u_values,
            (
                grid_x[missing_u],
                grid_y[missing_u]
            ),
            method="nearest"
        )

    missing_v = np.isnan(grid_v)

    if np.any(missing_v):

        grid_v[missing_v] = griddata(
            points,
            v_values,
            (
                grid_x[missing_v],
                grid_y[missing_v]
            ),
            method="nearest"
        )

    # --------------------------------------------------------
    # Smooth displacement field
    # --------------------------------------------------------

    smooth_u = gaussian_filter(
        grid_u,
        sigma=GAUSSIAN_SIGMA
    )

    smooth_v = gaussian_filter(
        grid_v,
        sigma=GAUSSIAN_SIGMA
    )

    # --------------------------------------------------------
    # Calculate displacement gradients
    # --------------------------------------------------------

    du_dy, du_dx = np.gradient(
        smooth_u,
        GRID_STEP,
        GRID_STEP
    )

    dv_dy, dv_dx = np.gradient(
        smooth_v,
        GRID_STEP,
        GRID_STEP
    )

    # --------------------------------------------------------
    # Strain components
    # --------------------------------------------------------

    epsilon_xx = du_dx

    epsilon_yy = dv_dy

    epsilon_xy = 0.5 * (
        du_dy + dv_dx
    )

    # --------------------------------------------------------
    # Principal strain
    # --------------------------------------------------------

    mean_strain = (
        epsilon_xx +
        epsilon_yy
    ) / 2

    radius = np.sqrt(
        (
            (epsilon_xx - epsilon_yy) / 2
        ) ** 2
        +
        epsilon_xy ** 2
    )

    principal_1 = mean_strain + radius

    principal_2 = mean_strain - radius

    max_principal = np.maximum(
        np.abs(principal_1),
        np.abs(principal_2)
    )

    return (
        grid_x,
        grid_y,
        smooth_u,
        smooth_v,
        max_principal
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    frame,
    image,
    grid_x,
    grid_y,
    strain,
    interval_name
):

    height, width = image.shape[:2]

    fixture_mask = create_fixture_mask(
        height,
        width
    )

    # --------------------------------------------------------
    # Mask fixture on grid
    # --------------------------------------------------------

    gx = grid_x.astype(int)
    gy = grid_y.astype(int)

    valid_x = (
        (gx >= 0) &
        (gx < width)
    )

    valid_y = (
        (gy >= 0) &
        (gy < height)
    )

    grid_fixture_mask = np.zeros(
        strain.shape,
        dtype=bool
    )

    valid = valid_x & valid_y

    grid_fixture_mask[valid] = (
        fixture_mask[
            gy[valid],
            gx[valid]
        ] == 1
    )

    display_strain = strain.copy()

    display_strain[
        grid_fixture_mask
    ] = np.nan

    # --------------------------------------------------------
    # Determine scale
    # --------------------------------------------------------

    valid_values = display_strain[
        np.isfinite(display_strain)
    ]

    if len(valid_values) == 0:
        print("No valid strain values.")
        return None

    vmax = np.percentile(
        valid_values,
        99
    )

    if vmax <= 0:
        vmax = np.max(valid_values)

    # --------------------------------------------------------
    # Find maximum outside fixture
    # --------------------------------------------------------

    outside_values = valid_values

    max_index = np.argmax(
        outside_values
    )

    maximum_strain = outside_values[
        max_index
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    heat = ax.pcolormesh(
        grid_x,
        grid_y,
        display_strain,
        cmap=DIC_CMAP,
        shading="auto",
        alpha=0.70,
        vmin=0,
        vmax=vmax
    )

    # --------------------------------------------------------
    # Draw fixture boundary
    # --------------------------------------------------------

    polygon_closed = np.vstack([
        FIXTURE_POLYGON,
        FIXTURE_POLYGON[0]
    ])

    ax.plot(
        polygon_closed[:, 0],
        polygon_closed[:, 1],
        "k-",
        linewidth=2
    )

    # --------------------------------------------------------
    # Color bar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        heat,
        ax=ax
    )

    cbar.set_label(
        "Maximum principal strain"
    )

    ax.set_title(
        f"DIC Strain Heatmap: {interval_name}"
    )

    ax.set_xlabel(
        "X (pixels)"
    )

    ax.set_ylabel(
        "Y (pixels)"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"strain_heatmap_{interval_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )

    return maximum_strain


# ============================================================
# PROCESS ONE INTERVAL
# ============================================================

def process_interval(
    start_frame,
    end_frame,
    image_map
):

    print("\n======================================")
    print(
        f"Processing {start_frame} -> {end_frame}"
    )
    print("======================================")

    csv_file = os.path.join(
        DIC_DIR,
        f"dic_{start_frame}_{end_frame}.csv"
    )

    if not os.path.exists(csv_file):

        print(
            "DIC file not found:",
            csv_file
        )

        return None

    if end_frame not in image_map:

        print(
            "End image not found:",
            end_frame
        )

        return None

    # --------------------------------------------------------
    # Read DIC CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_file
    )

    print(
        "Original DIC points:",
        len(df)
    )

    # --------------------------------------------------------
    # ZNCC filtering
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ].copy()

    print(
        "After ZNCC filtering:",
        len(df)
    )

    if len(df) < 100:

        print(
            "Too few valid DIC points."
        )

        return None

    # --------------------------------------------------------
    # Load end image
    # --------------------------------------------------------

    image_file = image_map[
        end_frame
    ]

    image = cv2.imread(
        image_file,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Could not read image:",
            image_file
        )

        return None

    height, width = image.shape

    # --------------------------------------------------------
    # Remove fixture BEFORE interpolation
    # --------------------------------------------------------

    df_clean = remove_fixture_points(
        df,
        height,
        width
    )

    print(
        "Fixture points removed:",
        len(df) - len(df_clean)
    )

    print(
        "Remaining concrete points:",
        len(df_clean)
    )

    # --------------------------------------------------------
    # Calculate strain
    # --------------------------------------------------------

    (
        grid_x,
        grid_y,
        smooth_u,
        smooth_v,
        strain
    ) = calculate_strain(
        df_clean,
        height,
        width
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_strain = strain[
        np.isfinite(strain)
    ]

    mean_strain = np.mean(
        valid_strain
    )

    p95 = np.percentile(
        valid_strain,
        95
    )

    p99 = np.percentile(
        valid_strain,
        99
    )

    maximum = np.max(
        valid_strain
    )

    max_position = np.unravel_index(
        np.argmax(strain),
        strain.shape
    )

    max_y = grid_y[
        max_position
    ]

    max_x = grid_x[
        max_position
    ]

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    interval_name = (
        f"{start_frame}_{end_frame}"
    )

    create_heatmap(
        end_frame,
        image,
        grid_x,
        grid_y,
        strain,
        interval_name
    )

    # --------------------------------------------------------
    # Save strain grid
    # --------------------------------------------------------

    strain_df = pd.DataFrame({
        "X": grid_x.flatten(),
        "Y": grid_y.flatten(),
        "Max_Principal_Strain": strain.flatten()
    })

    strain_csv = os.path.join(
        OUTPUT_DIR,
        f"strain_{interval_name}.csv"
    )

    strain_df.to_csv(
        strain_csv,
        index=False
    )

    print(
        "Saved strain CSV:",
        strain_csv
    )

    print("\nSTRAIN STATISTICS")

    print(
        "Mean:",
        mean_strain
    )

    print(
        "95th percentile:",
        p95
    )

    print(
        "99th percentile:",
        p99
    )

    print(
        "Maximum:",
        maximum
    )

    print(
        "Maximum location:",
        f"X={max_x:.1f}, Y={max_y:.1f}"
    )

    return {
        "Frame_Start": start_frame,
        "Frame_End": end_frame,
        "Original_DIC_Points": len(df),
        "Valid_DIC_Points": len(df_clean),
        "Mean_Strain": mean_strain,
        "P95_Strain": p95,
        "P99_Strain": p99,
        "Maximum_Strain": maximum,
        "Maximum_X": max_x,
        "Maximum_Y": max_y
    }


# ============================================================
# MAIN
# ============================================================

print("==============================================")
print("   STAGE 6F1 - FIXTURE MASKED DIC STRAIN")
print("==============================================")

image_map = load_image_map()

print(
    "\nAvailable ROI frames:",
    len(image_map)
)

print(
    "First available frame:",
    min(image_map.keys())
)

print(
    "Last available frame:",
    max(image_map.keys())
)


# ------------------------------------------------------------
# Save fixture polygon for documentation
# ------------------------------------------------------------

polygon_file = os.path.join(
    OUTPUT_DIR,
    "fixture_polygon.json"
)

with open(
    polygon_file,
    "w"
) as f:

    json.dump(
        FIXTURE_POLYGON.tolist(),
        f,
        indent=4
    )

print(
    "\nFixture polygon saved:",
    polygon_file
)


# ------------------------------------------------------------
# Process all intervals
# ------------------------------------------------------------

summary = []

for start_frame, end_frame in INTERVALS:

    result = process_interval(
        start_frame,
        end_frame,
        image_map
    )

    if result is not None:
        summary.append(result)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

if summary:

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "stage6F1_strain_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        "\n=============================================="
    )

    print(
        "SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved summary:",
        summary_file
    )


print(
    "\n=============================================="
)

print(
    "STAGE 6F1 COMPLETE"
)

print(
    "=============================================="
)