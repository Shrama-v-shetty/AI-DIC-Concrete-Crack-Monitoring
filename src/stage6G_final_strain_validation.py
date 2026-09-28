import os
import glob
import re

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
    "Stage6G_Final_Strains"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

GRID_STEP = 10

GAUSSIAN_SIGMA = 2

INTERVALS = [
    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432),
]


# ============================================================
# LOADING / FIXTURE ZONE
#
# This is used ONLY for interpretation/statistics.
#
# We deliberately do NOT remove these DIC points before
# interpolation because Stage 6F gave the better field.
# ============================================================

LOADING_X_MIN = 900
LOADING_X_MAX = 1080

LOADING_Y_MIN = 450
LOADING_Y_MAX = 638


# ============================================================
# ACTUAL BLACK FIXTURE POLYGON
#
# Used only to prevent the fixture itself from being coloured.
# ============================================================

FIXTURE_POLYGON = np.array([
    [955, 638],
    [1020, 638],
    [1010, 545],
    [1007, 525],
    [1000, 513],
    [990, 507],
    [978, 506],
    [968, 512],
    [963, 523],
    [960, 540],
    [958, 560]
], dtype=np.int32)


# ============================================================
# HEATMAP COLOUR MAP
#
# Purple -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red
# ============================================================

DIC_CMAP = LinearSegmentedColormap.from_list(
    "dic_heatmap",
    [
        "purple",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    ],
    N=256
)


# ============================================================
# FRAME NUMBER EXTRACTION
# ============================================================

def get_frame_number(filename):

    name = os.path.basename(filename)

    numbers = re.findall(r"\d+", name)

    if len(numbers) < 2:
        return None

    # Actual frame number is second-last number
    return int(numbers[-2])


# ============================================================
# LOAD IMAGE MAP
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
# CREATE FIXTURE MASK
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
# CREATE BROADER LOADING-ZONE MASK
#
# Used for statistics only.
# ============================================================

def create_loading_zone_mask(height, width):

    mask = np.zeros(
        (height, width),
        dtype=bool
    )

    x1 = max(0, LOADING_X_MIN)
    x2 = min(width, LOADING_X_MAX)

    y1 = max(0, LOADING_Y_MIN)
    y2 = min(height, LOADING_Y_MAX)

    mask[y1:y2, x1:x2] = True

    return mask


# ============================================================
# CALCULATE DIC STRAIN
#
# IMPORTANT:
# Same basic calculation as Stage 6F.
# Fixture points are NOT removed before interpolation.
# ============================================================

def calculate_strain(df, height, width):

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
    # Fill holes with nearest-neighbour interpolation
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
    # Gradients
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
    # Principal strains
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
# GET GRID LOADING MASK
# ============================================================

def get_grid_loading_mask(
    grid_x,
    grid_y
):

    return (
        (grid_x >= LOADING_X_MIN) &
        (grid_x <= LOADING_X_MAX) &
        (grid_y >= LOADING_Y_MIN) &
        (grid_y <= LOADING_Y_MAX)
    )


# ============================================================
# GET GRID FIXTURE MASK
# ============================================================

def get_grid_fixture_mask(
    grid_x,
    grid_y,
    height,
    width
):

    fixture_mask = create_fixture_mask(
        height,
        width
    )

    gx = np.clip(
        grid_x.astype(int),
        0,
        width - 1
    )

    gy = np.clip(
        grid_y.astype(int),
        0,
        height - 1
    )

    return (
        fixture_mask[gy, gx] == 1
    )


# ============================================================
# FIND HIGH-STRAIN REGION
# ============================================================

def find_high_strain_region(
    grid_x,
    grid_y,
    strain,
    height,
    width
):

    loading_mask = get_grid_loading_mask(
        grid_x,
        grid_y
    )

    fixture_mask = get_grid_fixture_mask(
        grid_x,
        grid_y,
        height,
        width
    )

    # Exclude loading region from analysis
    analysis_strain = strain.copy()

    analysis_strain[
        loading_mask
    ] = np.nan

    # Also exclude actual fixture
    analysis_strain[
        fixture_mask
    ] = np.nan

    valid = np.isfinite(
        analysis_strain
    )

    if not np.any(valid):
        return None

    valid_values = analysis_strain[
        valid
    ]

    threshold = np.percentile(
        valid_values,
        99
    )

    high_mask = (
        analysis_strain >= threshold
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    binary = (
        high_mask.astype(np.uint8)
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    regions = []

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < 20:
            continue

        component = (
            labels == label
        )

        component_values = analysis_strain[
            component
        ]

        component_max = np.nanmax(
            component_values
        )

        centroid_x = centroids[
            label,
            0
        ]

        centroid_y = centroids[
            label,
            1
        ]

        width_region = stats[
            label,
            cv2.CC_STAT_WIDTH
        ] * GRID_STEP

        height_region = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ] * GRID_STEP

        regions.append({
            "area": area,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width_region,
            "height": height_region,
            "max_strain": component_max
        })

    if not regions:
        return {
            "threshold": threshold,
            "area": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "width": 0,
            "height": 0,
            "max_strain": np.nan
        }

    # Largest meaningful high-strain region
    regions.sort(
        key=lambda r: r["area"],
        reverse=True
    )

    return {
        "threshold": threshold,
        **regions[0]
    }


# ============================================================
# CREATE FINAL HEATMAP
# ============================================================

def create_heatmap(
    image,
    grid_x,
    grid_y,
    strain,
    interval_name,
    height,
    width,
    region
):

    # --------------------------------------------------------
    # Copy strain field
    # --------------------------------------------------------

    display_strain = strain.copy()

    # Actual fixture only
    fixture_mask = get_grid_fixture_mask(
        grid_x,
        grid_y,
        height,
        width
    )

    display_strain[
        fixture_mask
    ] = np.nan

    valid_values = display_strain[
        np.isfinite(display_strain)
    ]

    vmax = np.percentile(
        valid_values,
        99
    )

    if vmax <= 0:
        vmax = np.max(
            valid_values
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(17, 7)
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
        alpha=0.68,
        vmin=0,
        vmax=vmax
    )

    # --------------------------------------------------------
    # Fixture outline
    # --------------------------------------------------------

    polygon = np.vstack([
        FIXTURE_POLYGON,
        FIXTURE_POLYGON[0]
    ])

    ax.plot(
        polygon[:, 0],
        polygon[:, 1],
        "k-",
        linewidth=2
    )

    # --------------------------------------------------------
    # Mark detected high-strain centroid
    # --------------------------------------------------------

    if region is not None:

        if np.isfinite(
            region["centroid_x"]
        ):

            ax.plot(
                region["centroid_x"],
                region["centroid_y"],
                "wo",
                markersize=7,
                markeredgecolor="black"
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
        f"DIC deformation/strain heatmap — {interval_name}"
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
        f"heatmap_{interval_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved heatmap:",
        output_file
    )


# ============================================================
# PROCESS INTERVAL
# ============================================================

def process_interval(
    start_frame,
    end_frame,
    image_map
):

    print("\n========================================")
    print(
        f"Processing {start_frame} -> {end_frame}"
    )
    print("========================================")

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
    # Read DIC
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_file
    )

    original_count = len(df)

    # --------------------------------------------------------
    # ZNCC filter
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ].copy()

    print(
        "Original DIC points:",
        original_count
    )

    print(
        "After ZNCC filtering:",
        len(df)
    )

    if len(df) < 100:

        print(
            "Too few DIC points."
        )

        return None

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_map[end_frame],
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Could not read image."
        )

        return None

    height, width = image.shape

    print(
        "Image size:",
        width,
        "x",
        height
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
        df,
        height,
        width
    )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    all_valid = strain[
        np.isfinite(strain)
    ]

    mean_strain = np.mean(
        all_valid
    )

    p95 = np.percentile(
        all_valid,
        95
    )

    p99 = np.percentile(
        all_valid,
        99
    )

    maximum = np.max(
        all_valid
    )

    # --------------------------------------------------------
    # High strain region OUTSIDE loading zone
    # --------------------------------------------------------

    region = find_high_strain_region(
        grid_x,
        grid_y,
        strain,
        height,
        width
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    interval_name = (
        f"{start_frame}_{end_frame}"
    )

    create_heatmap(
        image,
        grid_x,
        grid_y,
        strain,
        interval_name,
        height,
        width,
        region
    )

    # --------------------------------------------------------
    # Save strain grid
    # --------------------------------------------------------

    strain_df = pd.DataFrame({
        "X": grid_x.flatten(),
        "Y": grid_y.flatten(),
        "Max_Principal_Strain": strain.flatten()
    })

    strain_file = os.path.join(
        OUTPUT_DIR,
        f"strain_{interval_name}.csv"
    )

    strain_df.to_csv(
        strain_file,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nSTRAIN STATISTICS")

    print(
        f"Mean strain       : {mean_strain:.8f}"
    )

    print(
        f"95th percentile   : {p95:.8f}"
    )

    print(
        f"99th percentile   : {p99:.8f}"
    )

    print(
        f"Maximum strain    : {maximum:.8f}"
    )

    if region is not None:

        print("\nHIGH-STRAIN REGION")
        print(
            f"Threshold         : {region['threshold']:.8f}"
        )
        print(
            f"Area              : {region['area']} pixels"
        )
        print(
            f"Centroid X        : {region['centroid_x']:.2f}"
        )
        print(
            f"Centroid Y        : {region['centroid_y']:.2f}"
        )
        print(
            f"Width             : {region['width']}"
        )
        print(
            f"Height            : {region['height']}"
        )
        print(
            f"Region max strain : {region['max_strain']:.8f}"
        )

    return {
        "Frame_Start": start_frame,
        "Frame_End": end_frame,
        "Original_DIC_Points": original_count,
        "Valid_DIC_Points": len(df),
        "Mean_Strain": mean_strain,
        "P95_Strain": p95,
        "P99_Strain": p99,
        "Maximum_Strain": maximum,
        "Band_Threshold": (
            region["threshold"]
            if region else np.nan
        ),
        "Band_Area": (
            region["area"]
            if region else 0
        ),
        "Band_Centroid_X": (
            region["centroid_x"]
            if region else np.nan
        ),
        "Band_Centroid_Y": (
            region["centroid_y"]
            if region else np.nan
        ),
        "Band_Width": (
            region["width"]
            if region else 0
        ),
        "Band_Height": (
            region["height"]
            if region else 0
        ),
        "Band_Max_Strain": (
            region["max_strain"]
            if region else np.nan
        )
    }


# ============================================================
# MAIN
# ============================================================

print("==============================================")
print(" STAGE 6G - FINAL DIC STRAIN VALIDATION")
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

summary = []

for start_frame, end_frame in INTERVALS:

    result = process_interval(
        start_frame,
        end_frame,
        image_map
    )

    if result is not None:
        summary.append(result)


# ============================================================
# SAVE SUMMARY
# ============================================================

if summary:

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "stage6G_strain_band_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n==============================================")
    print("STAGE 6G SUMMARY")
    print("==============================================")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved summary:",
        summary_file
    )


print("\n==============================================")
print("STAGE 6G COMPLETE")
print("==============================================")