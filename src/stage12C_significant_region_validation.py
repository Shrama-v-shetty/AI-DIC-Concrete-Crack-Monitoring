import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# STAGE 12C - SIGNIFICANT REGION VALIDATION
# Region: 3000 -> 3432
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

IMAGE_DIR = os.path.join(BASE_DIR, "BGSL_ROI")
DIC_DIR = os.path.join(BASE_DIR, "Stage6C_Crack_Progression")

EXPERIMENTAL_XLS = os.path.join(
    os.path.dirname(BASE_DIR),
    "experimental_data",
    "combined_image_load_data_0_to_6600.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage12C_Significant_Region_Validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_1 = 3000
FRAME_2 = 3432

GRID_STEP = 10
GAUSSIAN_SIGMA = 2

# Loading/fixture area used only for statistical masking
FIXTURE_X1, FIXTURE_X2 = 930, 1080
FIXTURE_Y1, FIXTURE_Y2 = 450, 638


# ============================================================
# FIND FRAME NUMBER
# ============================================================

def extract_frame_number(filename):
    numbers = re.findall(r"\d+", os.path.basename(filename))

    if len(numbers) >= 2:
        # Example:
        # BGSL-2-3432_0.tif
        # numbers = ['2', '3432', '0']
        # actual frame = second-last number
        return int(numbers[-2])

    return None


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(frame):

    for name in os.listdir(IMAGE_DIR):

        if not name.lower().endswith(
            (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        ):
            continue

        if extract_frame_number(name) == frame:
            return os.path.join(IMAGE_DIR, name)

    return None


# ============================================================
# FIND STAGE 6C DIC FILE
# ============================================================

def find_dic_csv(frame1, frame2):

    path = os.path.join(
        DIC_DIR,
        f"dic_{frame1}_{frame2}.csv"
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# CALCULATE PRINCIPAL STRAIN
# ============================================================

def calculate_strain(df):

    rename = {}

    for column in df.columns:

        name = column.strip().lower()

        if name == "x":
            rename[column] = "X"

        elif name == "y":
            rename[column] = "Y"

        elif name == "dx":
            rename[column] = "dx"

        elif name == "dy":
            rename[column] = "dy"

    df = df.rename(columns=rename)

    required = ["X", "Y", "dx", "dy"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing DIC columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    x = pd.to_numeric(
        df["X"],
        errors="coerce"
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        df["Y"],
        errors="coerce"
    ).to_numpy(dtype=float)

    dx = pd.to_numeric(
        df["dx"],
        errors="coerce"
    ).to_numpy(dtype=float)

    dy = pd.to_numeric(
        df["dy"],
        errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(dx)
        & np.isfinite(dy)
    )

    x = x[valid]
    y = y[valid]
    dx = dx[valid]
    dy = dy[valid]

    if len(x) < 50:

        raise RuntimeError(
            f"Too few valid DIC points: {len(x)}"
        )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    x_values = np.arange(
        np.floor(x.min()),
        np.ceil(x.max()) + GRID_STEP,
        GRID_STEP
    )

    y_values = np.arange(
        np.floor(y.min()),
        np.ceil(y.max()) + GRID_STEP,
        GRID_STEP
    )

    gx, gy = np.meshgrid(
        x_values,
        y_values
    )

    # --------------------------------------------------------
    # INTERPOLATE DX / DY
    # --------------------------------------------------------

    gdx = griddata(
        (x, y),
        dx,
        (gx, gy),
        method="linear"
    )

    gdy = griddata(
        (x, y),
        dy,
        (gx, gy),
        method="linear"
    )

    # Fill interpolation gaps using nearest values
    for field, values in [
        (gdx, dx),
        (gdy, dy)
    ]:

        missing_mask = ~np.isfinite(field)

        if np.any(missing_mask):

            field[missing_mask] = griddata(
                (x, y),
                values,
                (gx[missing_mask], gy[missing_mask]),
                method="nearest"
            )

    # --------------------------------------------------------
    # SMOOTH
    # --------------------------------------------------------

    gdx = gaussian_filter(
        gdx,
        sigma=GAUSSIAN_SIGMA
    )

    gdy = gaussian_filter(
        gdy,
        sigma=GAUSSIAN_SIGMA
    )

    # --------------------------------------------------------
    # SPATIAL DERIVATIVES
    # --------------------------------------------------------

    du_dy, du_dx = np.gradient(
        gdx,
        GRID_STEP,
        GRID_STEP
    )

    dv_dy, dv_dx = np.gradient(
        gdy,
        GRID_STEP,
        GRID_STEP
    )

    # --------------------------------------------------------
    # STRAIN COMPONENTS
    # --------------------------------------------------------

    exx = du_dx
    eyy = dv_dy

    exy = 0.5 * (
        du_dy + dv_dx
    )

    # --------------------------------------------------------
    # MAXIMUM PRINCIPAL STRAIN
    # --------------------------------------------------------

    mean_strain = 0.5 * (
        exx + eyy
    )

    radius = np.sqrt(
        ((exx - eyy) * 0.5) ** 2
        + exy ** 2
    )

    principal_strain = (
        mean_strain + radius
    )

    return gx, gy, principal_strain


# ============================================================
# HEATMAP COLOR MAP
# ============================================================

def heatmap_colormap():

    return LinearSegmentedColormap.from_list(
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
# LOAD ORIGINAL EXPERIMENTAL EXCEL
# ============================================================

def load_experimental_data():

    if not os.path.exists(
        EXPERIMENTAL_XLS
    ):
        return None

    excel = pd.read_excel(
        EXPERIMENTAL_XLS
    )

    if "Image Number" not in excel.columns:

        raise RuntimeError(
            "Experimental Excel file does not contain "
            "'Image Number'."
        )

    excel["Image Number"] = pd.to_numeric(
        excel["Image Number"],
        errors="coerce"
    )

    return excel


# ============================================================
# GET EXPERIMENTAL ROW FOR A FRAME
# ============================================================

def get_endpoint_values(
    excel,
    frame
):

    if excel is None:
        return None

    rows = excel[
        excel["Image Number"] == frame
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# ============================================================
# SAFE NUMERIC VALUE
# ============================================================

def numeric_value(
    row,
    column
):

    if row is None:
        return np.nan

    if column not in row.index:
        return np.nan

    return pd.to_numeric(
        row[column],
        errors="coerce"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("STAGE 12C - SIGNIFICANT REGION VALIDATION")
print("=" * 65)

print(
    f"Pair: {FRAME_1} -> {FRAME_2}"
)

# ------------------------------------------------------------
# FIND FILES
# ------------------------------------------------------------

img1 = find_image(FRAME_1)
img2 = find_image(FRAME_2)
dic_path = find_dic_csv(
    FRAME_1,
    FRAME_2
)

print(
    "Frame 1 image:",
    img1
)

print(
    "Frame 2 image:",
    img2
)

print(
    "DIC CSV:",
    dic_path
)

if (
    img1 is None
    or img2 is None
    or dic_path is None
):

    raise FileNotFoundError(
        "Required image/DIC file was not found."
    )


# ------------------------------------------------------------
# READ END IMAGE
# ------------------------------------------------------------

image2 = cv2.imread(
    img2,
    cv2.IMREAD_GRAYSCALE
)

if image2 is None:

    raise RuntimeError(
        f"Could not read image:\n{img2}"
    )


# ------------------------------------------------------------
# READ DIC
# ------------------------------------------------------------

dic = pd.read_csv(
    dic_path
)

gx, gy, strain = calculate_strain(
    dic
)


# ============================================================
# STRAIN STATISTICS
# ============================================================

stat_mask = ~(
    (gx >= FIXTURE_X1)
    & (gx <= FIXTURE_X2)
    & (gy >= FIXTURE_Y1)
    & (gy <= FIXTURE_Y2)
)

valid = (
    np.isfinite(strain)
    & stat_mask
)

strain_values = strain[
    valid
]

if strain_values.size == 0:

    raise RuntimeError(
        "No valid strain cells remain after masking."
    )

mean_strain = float(
    np.mean(strain_values)
)

p95 = float(
    np.percentile(
        strain_values,
        95
    )
)

p99 = float(
    np.percentile(
        strain_values,
        99
    )
)

max_strain = float(
    np.max(strain_values)
)


# Highest 1%
high_mask = (
    valid
    & (strain >= p99)
)

high_y, high_x = np.where(
    high_mask
)

if len(high_x) > 0:

    centroid_x = float(
        np.mean(
            gx[high_y, high_x]
        )
    )

    centroid_y = float(
        np.mean(
            gy[high_y, high_x]
        )
    )

    high_x_min = float(
        np.min(
            gx[high_y, high_x]
        )
    )

    high_x_max = float(
        np.max(
            gx[high_y, high_x]
        )
    )

    high_y_min = float(
        np.min(
            gy[high_y, high_x]
        )
    )

    high_y_max = float(
        np.max(
            gy[high_y, high_x]
        )
    )

else:

    centroid_x = np.nan
    centroid_y = np.nan

    high_x_min = np.nan
    high_x_max = np.nan

    high_y_min = np.nan
    high_y_max = np.nan


print()
print("STRAIN STATISTICS")

print(
    f"Mean : {mean_strain:.6f}"
)

print(
    f"P95  : {p95:.6f}"
)

print(
    f"P99  : {p99:.6f}"
)

print(
    f"Max  : {max_strain:.6f}"
)


print()
print("HIGH-STRAIN REGION")

if len(high_x) > 0:

    print(
        f"Cells    : {len(high_x)}"
    )

    print(
        f"Centroid : "
        f"({centroid_x:.1f}, "
        f"{centroid_y:.1f})"
    )

    print(
        f"X range  : "
        f"{high_x_min:.1f} -> "
        f"{high_x_max:.1f}"
    )

    print(
        f"Y range  : "
        f"{high_y_min:.1f} -> "
        f"{high_y_max:.1f}"
    )

else:

    print(
        "No high-strain cells found."
    )


# ============================================================
# EXPERIMENTAL DATA
# ============================================================

print()
print("EXPERIMENTAL DATA")

excel = load_experimental_data()

if excel is None:

    print(
        "Experimental Excel file not found:"
    )

    print(
        EXPERIMENTAL_XLS
    )

    time1 = np.nan
    time2 = np.nan

    load1 = np.nan
    load2 = np.nan

    disp1 = np.nan
    disp2 = np.nan

else:

    row1 = get_endpoint_values(
        excel,
        FRAME_1
    )

    row2 = get_endpoint_values(
        excel,
        FRAME_2
    )

    time1 = numeric_value(
        row1,
        "Time_0"
    )

    time2 = numeric_value(
        row2,
        "Time_0"
    )

    load1 = numeric_value(
        row1,
        "Load ch 2 kN"
    )

    load2 = numeric_value(
        row2,
        "Load ch 2 kN"
    )

    disp1 = numeric_value(
        row1,
        "Displacement ch 7 µm"
    )

    disp2 = numeric_value(
        row2,
        "Displacement ch 7 µm"
    )

    print(
        f"Frame {FRAME_1} time          : "
        f"{time1}"
    )

    print(
        f"Frame {FRAME_2} time          : "
        f"{time2}"
    )

    print(
        f"Frame {FRAME_1} load (kN)     : "
        f"{load1}"
    )

    print(
        f"Frame {FRAME_2} load (kN)     : "
        f"{load2}"
    )

    print(
        f"Frame {FRAME_1} ch7 (µm)      : "
        f"{disp1}"
    )

    print(
        f"Frame {FRAME_2} ch7 (µm)      : "
        f"{disp2}"
    )

    if (
        np.isfinite(load1)
        and np.isfinite(load2)
    ):

        print(
            f"Load change (kN)              : "
            f"{load2 - load1:.6f}"
        )

    if (
        np.isfinite(disp1)
        and np.isfinite(disp2)
    ):

        print(
            f"Ch7 displacement change (µm) : "
            f"{disp2 - disp1:.6f}"
        )

    if (
        np.isfinite(time1)
        and np.isfinite(time2)
    ):

        print(
            f"Time difference (sec)         : "
            f"{time2 - time1:.6f}"
        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary = pd.DataFrame(
    [{
        "Frame_1": FRAME_1,
        "Frame_2": FRAME_2,

        "Time_1": time1,
        "Time_2": time2,

        "Load_1_kN": load1,
        "Load_2_kN": load2,

        "Load_Delta_kN": (
            load2 - load1
            if (
                np.isfinite(load1)
                and np.isfinite(load2)
            )
            else np.nan
        ),

        "Displacement_ch7_1_um": disp1,
        "Displacement_ch7_2_um": disp2,

        "Displacement_ch7_Delta_um": (
            disp2 - disp1
            if (
                np.isfinite(disp1)
                and np.isfinite(disp2)
            )
            else np.nan
        ),

        "Mean_Principal_Strain": mean_strain,
        "P95_Principal_Strain": p95,
        "P99_Principal_Strain": p99,
        "Max_Principal_Strain": max_strain,

        "High_Strain_Cells": len(high_x),

        "High_Strain_Centroid_X": centroid_x,
        "High_Strain_Centroid_Y": centroid_y,

        "High_Strain_X_Min": high_x_min,
        "High_Strain_X_Max": high_x_max,

        "High_Strain_Y_Min": high_y_min,
        "High_Strain_Y_Max": high_y_max
    }]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    f"summary_{FRAME_1}_{FRAME_2}.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE STRAIN FIELD
# ============================================================

strain_field = pd.DataFrame(
    {
        "X": gx.ravel(),
        "Y": gy.ravel(),
        "Principal_Strain": strain.ravel()
    }
)

strain_field_path = os.path.join(
    OUTPUT_DIR,
    f"strain_field_{FRAME_1}_{FRAME_2}.csv"
)

strain_field.to_csv(
    strain_field_path,
    index=False
)


# ============================================================
# HEATMAP
# ============================================================

cmap = heatmap_colormap()

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    image2,
    cmap="gray"
)

plt.imshow(
    strain,
    extent=[
        gx.min(),
        gx.max(),
        gy.max(),
        gy.min()
    ],
    cmap=cmap,
    alpha=0.58,
    vmin=0,
    vmax=p99
)

plt.colorbar(
    label="Principal strain"
)

# Fixture rectangle
plt.gca().add_patch(
    plt.Rectangle(
        (
            FIXTURE_X1,
            FIXTURE_Y1
        ),
        FIXTURE_X2 - FIXTURE_X1,
        FIXTURE_Y2 - FIXTURE_Y1,
        fill=False,
        linestyle="--",
        linewidth=2
    )
)

# High-strain centroid
if np.isfinite(centroid_x):

    plt.scatter(
        [centroid_x],
        [centroid_y],
        marker="x",
        s=100,
        linewidths=2
    )

plt.title(
    f"DIC Principal Strain Heatmap: "
    f"{FRAME_1} -> {FRAME_2}"
)

plt.xlabel(
    "Image X (pixels)"
)

plt.ylabel(
    "Image Y (pixels)"
)

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    f"heatmap_{FRAME_1}_{FRAME_2}.png"
)

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# THREE PANEL VALIDATION
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 6)
)


# ------------------------------------------------------------
# PANEL 1 - ORIGINAL
# ------------------------------------------------------------

axes[0].imshow(
    image2,
    cmap="gray"
)

axes[0].set_title(
    f"Original frame {FRAME_2}"
)

axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")


# ------------------------------------------------------------
# PANEL 2 - FULL HEATMAP
# ------------------------------------------------------------

axes[1].imshow(
    image2,
    cmap="gray"
)

axes[1].imshow(
    strain,
    extent=[
        gx.min(),
        gx.max(),
        gy.max(),
        gy.min()
    ],
    cmap=cmap,
    alpha=0.58,
    vmin=0,
    vmax=p99
)

axes[1].add_patch(
    plt.Rectangle(
        (
            FIXTURE_X1,
            FIXTURE_Y1
        ),
        FIXTURE_X2 - FIXTURE_X1,
        FIXTURE_Y2 - FIXTURE_Y1,
        fill=False,
        linestyle="--",
        linewidth=2
    )
)

if np.isfinite(centroid_x):

    axes[1].scatter(
        [centroid_x],
        [centroid_y],
        marker="x",
        s=100,
        linewidths=2
    )

axes[1].set_title(
    "DIC strain heatmap"
)

axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")


# ------------------------------------------------------------
# PANEL 3 - CENTRAL REGION
# ------------------------------------------------------------

axes[2].imshow(
    image2,
    cmap="gray"
)

axes[2].imshow(
    strain,
    extent=[
        gx.min(),
        gx.max(),
        gy.max(),
        gy.min()
    ],
    cmap=cmap,
    alpha=0.62,
    vmin=0,
    vmax=p99
)

axes[2].set_xlim(
    max(
        0,
        FIXTURE_X1 - 350
    ),
    min(
        image2.shape[1],
        FIXTURE_X2 + 350
    )
)

axes[2].set_ylim(
    min(
        image2.shape[0],
        FIXTURE_Y2 + 100
    ),
    max(
        0,
        FIXTURE_Y1 - 400
    )
)

axes[2].set_title(
    "Central deformation region"
)

axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")


# ============================================================
# COLORBAR
# ============================================================

scalar_map = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=0,
        vmax=p99
    )
)

scalar_map.set_array([])

fig.colorbar(
    scalar_map,
    ax=axes,
    shrink=0.8,
    label="Principal strain"
)


fig.suptitle(
    f"Stage 12C Validation: "
    f"{FRAME_1} -> {FRAME_2}"
)

# Explicit spacing avoids the previous tight_layout warning.
fig.subplots_adjust(
    left=0.04,
    right=0.92,
    bottom=0.10,
    top=0.88,
    wspace=0.20
)


validation_path = os.path.join(
    OUTPUT_DIR,
    f"validation_{FRAME_1}_{FRAME_2}.png"
)

fig.savefig(
    validation_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# OUTPUT
# ============================================================

print()
print("OUTPUTS")

print(
    heatmap_path
)

print(
    validation_path
)

print(
    summary_path
)

print(
    strain_field_path
)

print()
print("STAGE 12C COMPLETE")

print(
    "Review validation_3000_3432.png "
    "before finalizing Stage 12."
)