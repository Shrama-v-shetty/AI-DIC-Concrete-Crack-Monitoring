import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# STAGE 22B - CLEAN DIC HEATMAP VALIDATION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

IMAGE_DIR = os.path.join(BASE_DIR, "BGSL_ROI")

STAGE22_DIR = os.path.join(
    BASE_DIR,
    "Stage22_Rolling_Reference_Validation"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage22B_Clean_Heatmap"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# VALIDATION INTERVAL
# ------------------------------------------------------------

FRAME_1 = 1274
FRAME_2 = 1374


# ------------------------------------------------------------
# DIC SETTINGS
# ------------------------------------------------------------

MIN_ZNCC = 0.985

GRID_STEP = 10

DISPLACEMENT_SIGMA = 2.0

STRAIN_SIGMA = 1.5

MIN_COMPONENT_PIXELS = 12


# ------------------------------------------------------------
# FIXTURE / LOADING REGION
# ------------------------------------------------------------

FIXTURE_X1 = 900
FIXTURE_X2 = 1080

FIXTURE_Y1 = 450
FIXTURE_Y2 = 638


# ============================================================
# FRAME NUMBER
# ============================================================

def frame_number(path):

    name = os.path.basename(path)

    nums = re.findall(r"\d+", name)

    # BGSL-2-1274_0.tif
    # numbers = 2, 1274, 0
    # actual frame = second-last number

    if len(nums) >= 2:
        return int(nums[-2])

    return None


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(frame):

    candidates = []

    candidates.extend(
        glob.glob(os.path.join(IMAGE_DIR, "*.tif"))
    )

    candidates.extend(
        glob.glob(os.path.join(IMAGE_DIR, "*.tiff"))
    )

    for path in candidates:

        if frame_number(path) == frame:
            return path

    raise FileNotFoundError(
        f"\nCould not find ROI image for frame {frame}\n"
        f"Directory:\n{IMAGE_DIR}"
    )


# ============================================================
# CHECK WHETHER CSV IS POINT-LEVEL DIC
# ============================================================

def normalize_dic_columns(df):

    rename = {}

    for c in df.columns:

        lc = str(c).strip().lower()

        # Standard Stage 9 style
        if lc == "x":
            rename[c] = "X"

        elif lc == "y":
            rename[c] = "Y"

        elif lc == "dx":
            rename[c] = "dx"

        elif lc == "dy":
            rename[c] = "dy"

        elif lc in [
            "zncc",
            "score",
            "best_score",
            "correlation",
            "correlation_score"
        ]:
            rename[c] = "ZNCC"

        # Stage 19B style
        elif lc in ["cx", "center_x"]:
            rename[c] = "X"

        elif lc in ["cy", "center_y"]:
            rename[c] = "Y"

        elif lc in ["best_dx", "displacement_x"]:
            rename[c] = "dx"

        elif lc in ["best_dy", "displacement_y"]:
            rename[c] = "dy"

    return df.rename(columns=rename)


def is_point_level_dic_csv(path):

    filename = os.path.basename(path).lower()

    # Never use summary files for spatial heatmaps.
    if "summary" in filename:
        return False

    try:

        df = pd.read_csv(
            path,
            nrows=5
        )

    except Exception:

        return False

    df = normalize_dic_columns(df)

    required = ["X", "Y", "dx", "dy"]

    return all(
        column in df.columns
        for column in required
    )


# ============================================================
# CHECK FRAME PAIR
# ============================================================

def csv_contains_pair(path, frame1, frame2):

    try:

        df = pd.read_csv(
            path,
            nrows=3
        )

    except Exception:

        return False

    columns = {
        str(c).strip()
        for c in df.columns
    }

    if "Frame_1" not in columns:
        return False

    if "Frame_2" not in columns:
        return False

    try:

        a = int(df.iloc[0]["Frame_1"])
        b = int(df.iloc[0]["Frame_2"])

        return (
            a == frame1 and
            b == frame2
        )

    except Exception:

        return False


# ============================================================
# FIND POINT-LEVEL DIC CSV
# ============================================================

def find_dic_points_csv(frame1, frame2):

    print("\nSearching for point-level DIC data...")

    search_dirs = [

        # Stage 22 rolling-reference results
        STAGE22_DIR,

        # Existing Stage 6 interval results
        os.path.join(
            BASE_DIR,
            "Stage6C_Crack_Progression_DIC"
        ),

        # Stage 19B
        os.path.join(
            BASE_DIR,
            "Stage19B_DIC_Results"
        ),

        # Stage 9
        os.path.join(
            BASE_DIR,
            "Stage9_Full_Sequential_DIC"
        ),
    ]

    summary_files_found = []

    for directory in search_dirs:

        if not os.path.isdir(directory):
            continue

        csv_files = glob.glob(
            os.path.join(
                directory,
                "**",
                "*.csv"
            ),
            recursive=True
        )

        for path in csv_files:

            filename = os.path.basename(path).lower()

            if "summary" in filename:

                summary_files_found.append(path)

                continue

            # First check that it is actually a point-level file.

            if not is_point_level_dic_csv(path):
                continue

            # Then see whether it identifies the requested pair.

            if csv_contains_pair(
                path,
                frame1,
                frame2
            ):

                print(
                    f"Point-level DIC CSV found:\n{path}"
                )

                return path

            # Also accept filenames containing the pair.

            stem = os.path.splitext(
                filename
            )[0]

            pair_patterns = [

                f"{frame1}_{frame2}",
                f"{frame1}-{frame2}",
                f"{frame1}to{frame2}",
                f"{frame1}__{frame2}",
            ]

            if any(
                pattern in stem
                for pattern in pair_patterns
            ):

                print(
                    f"Point-level DIC CSV found:\n{path}"
                )

                return path

    print(
        "\nNo point-level DIC CSV was found for "
        f"{frame1} -> {frame2}."
    )

    print(
        "\nThe following summary files may exist, "
        "but they cannot be used to create a spatial heatmap:"
    )

    for path in summary_files_found[:10]:

        print(
            "  ",
            path
        )

    return None


# ============================================================
# LOAD POINT-LEVEL DIC
# ============================================================

def load_dic_points(csv_path):

    print(
        f"\nLoading point-level DIC:\n{csv_path}"
    )

    df = pd.read_csv(csv_path)

    df = normalize_dic_columns(df)

    required = [
        "X",
        "Y",
        "dx",
        "dy"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nPoint-level DIC CSV is missing:\n"
            + ", ".join(missing)
            + "\n\nColumns found:\n"
            + ", ".join(
                map(str, df.columns)
            )
        )

    if "ZNCC" not in df.columns:

        print(
            "ZNCC column not found."
        )

        print(
            "Keeping all DIC points."
        )

        df["ZNCC"] = 1.0

    for column in required + ["ZNCC"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=required + ["ZNCC"]
    )

    before = len(df)

    df = df[
        df["ZNCC"] >= MIN_ZNCC
    ].copy()

    after = len(df)

    print(
        f"Valid DIC points before ZNCC filter : {before}"
    )

    print(
        f"Valid DIC points after ZNCC filter  : {after}"
    )

    if len(df) == 0:

        raise ValueError(
            "No DIC points remain after ZNCC filtering."
        )

    return df


# ============================================================
# BUILD GRID
# ============================================================

def make_grid(
    df,
    value_column,
    gx,
    gy
):

    points = np.column_stack(
        [
            df["X"].values,
            df["Y"].values
        ]
    )

    values = df[value_column].values

    grid = griddata(
        points,
        values,
        (gx, gy),
        method="linear"
    )

    bad = ~np.isfinite(grid)

    if np.any(bad):

        nearest = griddata(
            points,
            values,
            (gx, gy),
            method="nearest"
        )

        grid[bad] = nearest[bad]

    return grid


# ============================================================
# PRINCIPAL STRAIN
# ============================================================

def calculate_strain(df):

    x_min = int(
        np.floor(
            df["X"].min()
        )
    )

    x_max = int(
        np.ceil(
            df["X"].max()
        )
    )

    y_min = int(
        np.floor(
            df["Y"].min()
        )
    )

    y_max = int(
        np.ceil(
            df["Y"].max()
        )
    )

    gx_values = np.arange(
        x_min,
        x_max + GRID_STEP,
        GRID_STEP
    )

    gy_values = np.arange(
        y_min,
        y_max + GRID_STEP,
        GRID_STEP
    )

    gx, gy = np.meshgrid(
        gx_values,
        gy_values
    )

    dx = make_grid(
        df,
        "dx",
        gx,
        gy
    )

    dy = make_grid(
        df,
        "dy",
        gx,
        gy
    )

    # --------------------------------------------------------
    # Smooth displacement
    # --------------------------------------------------------

    dx = gaussian_filter(
        dx,
        sigma=DISPLACEMENT_SIGMA
    )

    dy = gaussian_filter(
        dy,
        sigma=DISPLACEMENT_SIGMA
    )

    # --------------------------------------------------------
    # Displacement gradients
    # --------------------------------------------------------

    du_dx = np.gradient(
        dx,
        GRID_STEP,
        axis=1
    )

    du_dy = np.gradient(
        dx,
        GRID_STEP,
        axis=0
    )

    dv_dx = np.gradient(
        dy,
        GRID_STEP,
        axis=1
    )

    dv_dy = np.gradient(
        dy,
        GRID_STEP,
        axis=0
    )

    # --------------------------------------------------------
    # Strain tensor
    # --------------------------------------------------------

    exx = du_dx

    eyy = dv_dy

    exy = 0.5 * (
        du_dy + dv_dx
    )

    # --------------------------------------------------------
    # Principal strain
    # --------------------------------------------------------

    mean = 0.5 * (
        exx + eyy
    )

    radius = np.sqrt(
        (
            0.5 *
            (exx - eyy)
        ) ** 2
        +
        exy ** 2
    )

    e1 = mean + radius

    e2 = mean - radius

    principal = np.maximum(
        np.abs(e1),
        np.abs(e2)
    )

    # --------------------------------------------------------
    # Final smoothing
    # --------------------------------------------------------

    principal = gaussian_filter(
        principal,
        sigma=STRAIN_SIGMA
    )

    return gx, gy, principal


# ============================================================
# FIXTURE MASK
# ============================================================

def apply_fixture_mask(
    field,
    gx,
    gy
):

    mask = (

        (gx >= FIXTURE_X1)
        &
        (gx <= FIXTURE_X2)
        &
        (gy >= FIXTURE_Y1)
        &
        (gy <= FIXTURE_Y2)

    )

    result = field.copy()

    result[mask] = np.nan

    return result


# ============================================================
# CLEAN HIGH-STRAIN REGION
# ============================================================

def find_high_strain_region(field):

    finite = np.isfinite(field)

    if not np.any(finite):

        return np.zeros_like(
            field,
            dtype=bool
        )

    values = field[finite]

    threshold = np.percentile(
        values,
        97.5
    )

    candidate = (
        finite &
        (field >= threshold)
    )

    candidate = binary_opening(
        candidate,
        iterations=1
    )

    candidate = binary_closing(
        candidate,
        iterations=2
    )

    labels, count = label(
        candidate
    )

    cleaned = np.zeros_like(
        candidate,
        dtype=bool
    )

    for idx in range(
        1,
        count + 1
    ):

        region = (
            labels == idx
        )

        if np.sum(region) >= MIN_COMPONENT_PIXELS:

            cleaned |= region

    return cleaned


# ============================================================
# COLORMAP
# ============================================================

def dic_colormap():

    return LinearSegmentedColormap.from_list(

        "dic_clean",

        [

            "#6A1B9A",  # purple
            "#283593",  # blue
            "#00BCD4",  # cyan
            "#4CAF50",  # green
            "#FFEB3B",  # yellow
            "#FF9800",  # orange
            "#D32F2F",  # red

        ]

    )


# ============================================================
# MAIN
# ============================================================

print("=" * 75)

print(
    "STAGE 22B - CLEAN DIC HEATMAP VALIDATION"
)

print("=" * 75)


print(
    f"\nFrames: {FRAME_1} -> {FRAME_2}"
)

print(
    f"\nROI image directory:\n{IMAGE_DIR}"
)


# ------------------------------------------------------------
# Images
# ------------------------------------------------------------

image1_path = find_image(
    FRAME_1
)

image2_path = find_image(
    FRAME_2
)

print(
    f"\nReference image: "
    f"{os.path.basename(image1_path)}"
)

print(
    f"End image      : "
    f"{os.path.basename(image2_path)}"
)


# ------------------------------------------------------------
# Point-level DIC CSV
# ------------------------------------------------------------

dic_csv = find_dic_points_csv(
    FRAME_1,
    FRAME_2
)

if dic_csv is None:

    print(
        "\n======================================================"
    )

    print(
        "POINT-LEVEL DIC DATA NOT FOUND"
    )

    print(
        "======================================================"
    )

    print(
        "\nThe rolling-reference summary CSV cannot be used "
        "for a spatial heatmap."
    )

    print(
        "\nWe need the actual point-level DIC result containing:"
    )

    print(
        "X, Y, dx, dy"
    )

    print(
        "\nNo heatmap was generated."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# Load DIC
# ------------------------------------------------------------

df = load_dic_points(
    dic_csv
)

print(
    f"\nAverage ZNCC: "
    f"{df['ZNCC'].mean():.6f}"
)


# ------------------------------------------------------------
# Read end image
# ------------------------------------------------------------

img = cv2.imread(
    image2_path,
    cv2.IMREAD_GRAYSCALE
)

if img is None:

    raise RuntimeError(
        "Could not read end image."
    )


# ------------------------------------------------------------
# Calculate strain
# ------------------------------------------------------------

print(
    "\nCalculating DIC principal strain..."
)

gx, gy, strain = calculate_strain(
    df
)


# ------------------------------------------------------------
# Fixture mask
# ------------------------------------------------------------

strain_masked = apply_fixture_mask(
    strain,
    gx,
    gy
)


# ------------------------------------------------------------
# High strain region
# ------------------------------------------------------------

high_region = find_high_strain_region(
    strain_masked
)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

finite = strain_masked[
    np.isfinite(
        strain_masked
    )
]

if len(finite) == 0:

    raise RuntimeError(
        "No valid strain values remain."
    )


p5 = np.percentile(
    finite,
    5
)

p95 = np.percentile(
    finite,
    95
)

p99 = np.percentile(
    finite,
    99
)

vmin = float(p5)

vmax = float(p99)

if vmax <= vmin:

    vmax = vmin + 1e-6


# ------------------------------------------------------------
# High strain centroid
# ------------------------------------------------------------

yy, xx = np.where(
    high_region
)

if len(xx) > 0:

    centroid_x = float(
        np.mean(
            gx[yy, xx]
        )
    )

    centroid_y = float(
        np.mean(
            gy[yy, xx]
        )
    )

else:

    centroid_x = np.nan

    centroid_y = np.nan


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print(
    "\n======================================================"
)

print(
    "STRAIN VALIDATION RESULTS"
)

print(
    "======================================================"
)

print(
    f"Valid DIC points : {len(df)}"
)

print(
    f"Average ZNCC     : "
    f"{df['ZNCC'].mean():.6f}"
)

print(
    f"5th percentile   : {p5:.6f}"
)

print(
    f"95th percentile  : {p95:.6f}"
)

print(
    f"99th percentile  : {p99:.6f}"
)

if not np.isnan(centroid_x):

    print(
        "\nHigh-strain region:"
    )

    print(
        f"Centroid X : {centroid_x:.1f}"
    )

    print(
        f"Centroid Y : {centroid_y:.1f}"
    )

    print(
        f"Grid cells : {len(xx)}"
    )

else:

    print(
        "\nNo coherent high-strain region detected."
    )


# ============================================================
# SAVE STRAIN FIELD
# ============================================================

field_df = pd.DataFrame({

    "X":
        gx.ravel(),

    "Y":
        gy.ravel(),

    "Principal_Strain":
        strain_masked.ravel(),

    "High_Strain_Region":
        high_region.ravel().astype(int)

})


field_csv = os.path.join(

    OUTPUT_DIR,

    f"clean_strain_field_"
    f"{FRAME_1}_{FRAME_2}.csv"

)


field_df.to_csv(
    field_csv,
    index=False
)


# ============================================================
# CREATE HEATMAP
# ============================================================

print(
    "\nGenerating heatmap..."
)


fig, ax = plt.subplots(
    figsize=(15, 7)
)


# Original concrete image
ax.imshow(
    img,
    cmap="gray",
    extent=[
        0,
        img.shape[1],
        img.shape[0],
        0
    ]
)


# DIC strain heatmap
heat = ax.imshow(

    strain_masked,

    cmap=dic_colormap(),

    alpha=0.70,

    extent=[
        gx.min(),
        gx.max(),
        gy.max(),
        gy.min()
    ],

    vmin=vmin,

    vmax=vmax,

    interpolation="bilinear"

)


# High strain region outline
if len(xx) > 0:

    ax.contour(

        gx,
        gy,
        high_region.astype(float),

        levels=[0.5],

        linewidths=2.0,

        colors="white"

    )


ax.set_title(
    f"DIC Principal-Strain Heatmap — "
    f"{FRAME_1} → {FRAME_2}"
)

ax.set_xlabel(
    "Image X (pixels)"
)

ax.set_ylabel(
    "Image Y (pixels)"
)


cbar = fig.colorbar(
    heat,
    ax=ax
)

cbar.set_label(
    "Principal strain"
)


plt.tight_layout()


heatmap_path = os.path.join(

    OUTPUT_DIR,

    f"clean_heatmap_"
    f"{FRAME_1}_{FRAME_2}.png"

)


plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VALIDATION FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)


# Original
axes[0].imshow(
    img,
    cmap="gray"
)

axes[0].set_title(
    f"Concrete Image — Frame {FRAME_2}"
)

axes[0].axis(
    "off"
)


# Heatmap
axes[1].imshow(
    img,
    cmap="gray"
)

heat2 = axes[1].imshow(

    strain_masked,

    cmap=dic_colormap(),

    alpha=0.70,

    extent=[
        gx.min(),
        gx.max(),
        gy.max(),
        gy.min()
    ],

    vmin=vmin,

    vmax=vmax,

    interpolation="bilinear"

)


if len(xx) > 0:

    axes[1].contour(

        gx,
        gy,
        high_region.astype(float),

        levels=[0.5],

        colors="white",

        linewidths=2

    )


axes[1].set_title(
    f"DIC Principal Strain — "
    f"{FRAME_1} → {FRAME_2}"
)

axes[1].set_xlabel(
    "Image X (pixels)"
)

axes[1].set_ylabel(
    "Image Y (pixels)"
)


fig.colorbar(
    heat2,
    ax=axes[1],
    label="Principal strain"
)


plt.tight_layout()


validation_path = os.path.join(

    OUTPUT_DIR,

    f"validation_"
    f"{FRAME_1}_{FRAME_2}.png"

)


plt.savefig(
    validation_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "Frame_1":
        FRAME_1,

    "Frame_2":
        FRAME_2,

    "Valid_DIC_Points":
        len(df),

    "Average_ZNCC":
        df["ZNCC"].mean(),

    "Principal_Strain_P5":
        p5,

    "Principal_Strain_P95":
        p95,

    "Principal_Strain_P99":
        p99,

    "High_Strain_Centroid_X":
        centroid_x,

    "High_Strain_Centroid_Y":
        centroid_y,

    "High_Strain_Cells":
        len(xx),

}])


summary_path = os.path.join(

    OUTPUT_DIR,

    "stage22B_summary.csv"

)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n======================================================"
)

print(
    "STAGE 22B COMPLETE"
)

print(
    "======================================================"
)

print(
    f"\nHeatmap:"
)

print(
    heatmap_path
)

print(
    f"\nValidation figure:"
)

print(
    validation_path
)

print(
    f"\nStrain field:"
)

print(
    field_csv
)

print(
    f"\nSummary:"
)

print(
    summary_path
)

print(
    "\nIMPORTANT:"
)

print(
    "The highlighted high-strain region is a "
    "DIC deformation/strain indication."
)

print(
    "It should not be automatically labelled as "
    "a crack without image verification."
)