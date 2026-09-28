import cv2
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, find_objects
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS
# ============================================================

ROI_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

DIC_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage6C_Crack_Progression"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage6F_Strain_Band_Analysis"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# INTERVALS
# ============================================================

STAGES = [
    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432)
]


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

GRID_STEP = 10

GAUSSIAN_SIGMA = 2

# Minimum number of grid pixels for a region
MIN_REGION_PIXELS = 20


# ============================================================
# LOADING / FIXTURE REGION
#
# This region is NOT removed from the visualization.
# It is excluded when calculating the main localization region.
# ============================================================

FIXTURE_X_MIN = 900
FIXTURE_X_MAX = 1080

FIXTURE_Y_MIN = 450
FIXTURE_Y_MAX = 638


# ============================================================
# HEATMAP COLORS
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
# FRAME NUMBER
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(
        r"\d+",
        filename
    )

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# BUILD IMAGE MAP
# ============================================================

def build_image_map():

    files = glob.glob(
        os.path.join(
            ROI_FOLDER,
            "*.tif"
        )
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
# CALCULATE PRINCIPAL STRAIN
# ============================================================

def calculate_strain(csv_path):

    df = pd.read_csv(
        csv_path
    )

    print(
        "\nReading:",
        os.path.basename(csv_path)
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
        "Points after ZNCC filtering:",
        len(df)
    )


    if len(df) < 100:

        print(
            "WARNING: Too few DIC points."
        )

        return None


    x = df["X"].values
    y = df["Y"].values

    u = df["dx"].values
    v = df["dy"].values


    # --------------------------------------------------------
    # Grid
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
    # Interpolation
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
    # Fill holes
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
    # Displacement gradients
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
    # Principal strain
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


    principal_strain = np.maximum(
        np.abs(principal_1),
        np.abs(principal_2)
    )


    return X, Y, principal_strain


# ============================================================
# ANALYZE HIGH-STRAIN REGION
# ============================================================

def analyze_strain_band(
    X,
    Y,
    strain
):

    finite_mask = np.isfinite(
        strain
    )

    finite_values = strain[
        finite_mask
    ]


    if len(finite_values) == 0:
        return None


    # --------------------------------------------------------
    # Basic strain statistics
    # --------------------------------------------------------

    p95 = np.percentile(
        finite_values,
        95
    )

    p99 = np.percentile(
        finite_values,
        99
    )

    maximum = np.max(
        finite_values
    )

    mean_value = np.mean(
        finite_values
    )


    # --------------------------------------------------------
    # Exclude loading/fixture zone
    # --------------------------------------------------------

    fixture_mask = (
        (X >= FIXTURE_X_MIN) &
        (X <= FIXTURE_X_MAX) &
        (Y >= FIXTURE_Y_MIN) &
        (Y <= FIXTURE_Y_MAX)
    )


    analysis_mask = (
        finite_mask &
        (~fixture_mask)
    )


    analysis_values = strain[
        analysis_mask
    ]


    if len(analysis_values) == 0:
        return None


    analysis_p95 = np.percentile(
        analysis_values,
        95
    )

    analysis_p99 = np.percentile(
        analysis_values,
        99
    )


    analysis_max = np.max(
        analysis_values
    )


    # --------------------------------------------------------
    # High-strain threshold
    #
    # We use the 99th percentile outside the fixture.
    # This identifies the most localized deformation.
    # --------------------------------------------------------

    threshold = analysis_p99


    high_strain_mask = (
        analysis_mask &
        (strain >= threshold)
    )


    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=np.int8
    )

    labeled, number = label(
        high_strain_mask,
        structure=structure
    )


    if number == 0:

        return {
            "mean": mean_value,
            "p95": p95,
            "p99": p99,
            "maximum": maximum,
            "analysis_p95": analysis_p95,
            "analysis_p99": analysis_p99,
            "analysis_max": analysis_max,
            "threshold": threshold,
            "region_pixels": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "width": np.nan,
            "height": np.nan,
            "max_region_strain": np.nan,
            "localization_ratio": np.nan,
            "region_mask": high_strain_mask
        }


    # --------------------------------------------------------
    # Find connected components
    # --------------------------------------------------------

    components = []

    slices = find_objects(
        labeled
    )


    for component_id, component_slice in enumerate(
        slices,
        start=1
    ):

        if component_slice is None:
            continue


        component_mask = (
            labeled == component_id
        )

        pixel_count = np.sum(
            component_mask
        )


        if pixel_count < MIN_REGION_PIXELS:
            continue


        component_values = strain[
            component_mask
        ]


        component_max = np.max(
            component_values
        )


        component_mean = np.mean(
            component_values
        )


        component_x = X[
            component_mask
        ]

        component_y = Y[
            component_mask
        ]


        centroid_x = np.mean(
            component_x
        )

        centroid_y = np.mean(
            component_y
        )


        width = (
            np.max(component_x)
            -
            np.min(component_x)
        )


        height = (
            np.max(component_y)
            -
            np.min(component_y)
        )


        components.append({
            "id": component_id,
            "pixels": pixel_count,
            "max": component_max,
            "mean": component_mean,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "width": width,
            "height": height
        })


    # --------------------------------------------------------
    # No sufficiently large component
    # --------------------------------------------------------

    if len(components) == 0:

        return {
            "mean": mean_value,
            "p95": p95,
            "p99": p99,
            "maximum": maximum,
            "analysis_p95": analysis_p95,
            "analysis_p99": analysis_p99,
            "analysis_max": analysis_max,
            "threshold": threshold,
            "region_pixels": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "width": np.nan,
            "height": np.nan,
            "max_region_strain": np.nan,
            "localization_ratio": np.nan,
            "region_mask": high_strain_mask
        }


    # --------------------------------------------------------
    # Select most significant component
    #
    # Priority:
    # 1. Larger area
    # 2. Higher strain
    #
    # This avoids selecting one isolated noisy pixel.
    # --------------------------------------------------------

    components.sort(
        key=lambda item: (
            item["pixels"],
            item["max"]
        ),
        reverse=True
    )


    best = components[0]


    # --------------------------------------------------------
    # Localization ratio
    #
    # Fraction of analysis pixels belonging to the
    # detected high-strain region.
    # --------------------------------------------------------

    localization_ratio = (
        best["pixels"]
        /
        np.sum(analysis_mask)
    )


    # --------------------------------------------------------
    # Build selected component mask
    # --------------------------------------------------------

    selected_mask = (
        labeled == best["id"]
    )


    return {
        "mean": mean_value,
        "p95": p95,
        "p99": p99,
        "maximum": maximum,
        "analysis_p95": analysis_p95,
        "analysis_p99": analysis_p99,
        "analysis_max": analysis_max,
        "threshold": threshold,
        "region_pixels": best["pixels"],
        "centroid_x": best["centroid_x"],
        "centroid_y": best["centroid_y"],
        "width": best["width"],
        "height": best["height"],
        "max_region_strain": best["max"],
        "localization_ratio": localization_ratio,
        "region_mask": selected_mask
    }


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    frame_start,
    frame_end,
    image_path,
    X,
    Y,
    strain,
    analysis
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

        return


    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )


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


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )


    # ========================================================
    # ORIGINAL
    # ========================================================

    axes[0].imshow(
        image_rgb
    )

    axes[0].set_title(
        f"Original Image\nFrame {frame_end}"
    )

    axes[0].set_xlabel(
        "X coordinate (ROI pixels)"
    )

    axes[0].set_ylabel(
        "Y coordinate (ROI pixels)"
    )


    # Loading zone

    rect = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linestyle="--",
        linewidth=2
    )

    axes[0].add_patch(
        rect
    )


    # ========================================================
    # HEATMAP
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
        f"DIC Principal Strain\n"
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


    # Loading zone

    rect2 = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linestyle="--",
        linewidth=2
    )

    axes[1].add_patch(
        rect2
    )


    # ========================================================
    # HIGH-STRAIN REGION
    # ========================================================

    axes[2].imshow(
        image_rgb
    )


    axes[2].pcolormesh(
        X,
        Y,
        strain,
        cmap=dic_cmap,
        shading="auto",
        alpha=0.55,
        vmin=0,
        vmax=vmax
    )


    # Selected region

    selected_mask = analysis[
        "region_mask"
    ]


    region_overlay = np.where(
        selected_mask,
        strain,
        np.nan
    )


    axes[2].pcolormesh(
        X,
        Y,
        region_overlay,
        cmap="autumn",
        shading="auto",
        alpha=0.85
    )


    # --------------------------------------------------------
    # Region centroid
    # --------------------------------------------------------

    if np.isfinite(
        analysis["centroid_x"]
    ):

        axes[2].scatter(
            analysis["centroid_x"],
            analysis["centroid_y"],
            marker="x",
            s=100,
            linewidths=3
        )


        axes[2].text(
            analysis["centroid_x"] + 20,
            analysis["centroid_y"],
            "High-strain centroid",
            fontsize=9
        )


    # Loading zone

    rect3 = plt.Rectangle(
        (
            FIXTURE_X_MIN,
            FIXTURE_Y_MIN
        ),
        FIXTURE_X_MAX - FIXTURE_X_MIN,
        FIXTURE_Y_MAX - FIXTURE_Y_MIN,
        fill=False,
        linestyle="--",
        linewidth=2
    )

    axes[2].add_patch(
        rect3
    )


    # Zoom central region

    axes[2].set_xlim(
        650,
        1250
    )

    axes[2].set_ylim(
        600,
        150
    )


    axes[2].set_title(
        "Detected High-Strain Localization"
    )


    axes[2].set_xlabel(
        "X coordinate (ROI pixels)"
    )

    axes[2].set_ylabel(
        "Y coordinate (ROI pixels)"
    )


    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"Stage 6F — Automatic Strain Localization\n"
        f"Frame {frame_start} → {frame_end}",
        fontsize=14
    )


    plt.tight_layout()


    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"strain_band_{frame_start}_{frame_end}.png"
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


# ============================================================
# MAIN
# ============================================================

print(
    "================================================"
)

print(
    " STAGE 6F — AUTOMATIC STRAIN-BAND ANALYSIS"
)

print(
    "================================================"
)


image_map = build_image_map()


print(
    "\nAvailable ROI frames:",
    len(image_map)
)


summary = []


for frame_start, frame_end in STAGES:

    print(
        "\n================================================"
    )

    print(
        f"Processing {frame_start} -> {frame_end}"
    )

    print(
        "================================================"
    )


    # --------------------------------------------------------
    # Check image
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


    if not os.path.exists(
        csv_path
    ):

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
    # Analyze localization
    # --------------------------------------------------------

    analysis = analyze_strain_band(
        X,
        Y,
        strain
    )


    if analysis is None:

        print(
            "Could not analyze strain band."
        )

        continue


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nSTRAIN LOCALIZATION RESULT"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Mean strain:",
        analysis["mean"]
    )

    print(
        "95th percentile:",
        analysis["p95"]
    )

    print(
        "99th percentile:",
        analysis["p99"]
    )

    print(
        "Maximum strain:",
        analysis["maximum"]
    )

    print(
        "\nOutside loading zone:"
    )

    print(
        "95th percentile:",
        analysis["analysis_p95"]
    )

    print(
        "99th percentile:",
        analysis["analysis_p99"]
    )

    print(
        "Maximum:",
        analysis["analysis_max"]
    )

    print(
        "\nDetected high-strain region:"
    )

    print(
        "Pixels:",
        analysis["region_pixels"]
    )

    print(
        "Centroid X:",
        analysis["centroid_x"]
    )

    print(
        "Centroid Y:",
        analysis["centroid_y"]
    )

    print(
        "Width:",
        analysis["width"]
    )

    print(
        "Height:",
        analysis["height"]
    )

    print(
        "Maximum region strain:",
        analysis["max_region_strain"]
    )

    print(
        "Localization ratio:",
        analysis["localization_ratio"]
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    create_visualization(
        frame_start,
        frame_end,
        image_map[frame_end],
        X,
        Y,
        strain,
        analysis
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.append([
        frame_start,
        frame_end,
        analysis["mean"],
        analysis["p95"],
        analysis["p99"],
        analysis["maximum"],
        analysis["analysis_p95"],
        analysis["analysis_p99"],
        analysis["analysis_max"],
        analysis["region_pixels"],
        analysis["centroid_x"],
        analysis["centroid_y"],
        analysis["width"],
        analysis["height"],
        analysis["max_region_strain"],
        analysis["localization_ratio"]
    ])


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

if len(summary) > 0:

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Frame_Start",
            "Frame_End",
            "Mean_Strain",
            "95th_Percentile",
            "99th_Percentile",
            "Maximum_Strain",
            "Outside_Loading_95th",
            "Outside_Loading_99th",
            "Outside_Loading_Max",
            "High_Strain_Region_Pixels",
            "Centroid_X",
            "Centroid_Y",
            "Region_Width",
            "Region_Height",
            "Region_Max_Strain",
            "Localization_Ratio"
        ]
    )


    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "stage6F_strain_band_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print(
        "\n================================================"
    )

    print(
        " STAGE 6F COMPLETE"
    )

    print(
        "================================================"
    )


    print(
        "\nFINAL PROGRESSION SUMMARY"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


    print(
        "\nSaved summary:"
    )

    print(
        summary_path
    )


else:

    print(
        "\nNo results generated."
    )


print(
    "\n================================================"
)

print(
    " END OF STAGE 6F"
)

print(
    "================================================"
)