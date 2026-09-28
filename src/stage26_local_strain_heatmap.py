import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset"

DIC_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage24_Rolling_Reference_Heatmap",
    "dic_1274_1374.csv"
)

IMAGE_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "BGSL_ROI",
    "BGSL-2-1374_0.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage26_Local_Affine_Strain"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_HEATMAP = os.path.join(
    OUTPUT_DIR,
    "local_affine_heatmap_1274_1374.png"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "local_affine_strain_1274_1374.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.95

# Number of neighboring DIC points used for each
# local deformation estimate.
NEIGHBORS = 25

# Maximum neighborhood radius in pixels.
MAX_RADIUS = 100

# Grid spacing for final visualization.
GRID_STEP = 5

# Small smoothing only after local strain calculation.
# This is NOT used to create the strain itself.
VISUAL_SMOOTHING = 1.2


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("STAGE 26 - LOCAL AFFINE DIC STRAIN")
print("==============================================")


# ============================================================
# LOAD DIC
# ============================================================

print("\nLoading point-level DIC...")

df = pd.read_csv(DIC_FILE)

required = ["X", "Y", "dx", "dy", "ZNCC"]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

print("Original DIC points:", len(df))


# ============================================================
# QUALITY FILTER
# ============================================================

df = df[
    np.isfinite(df["X"]) &
    np.isfinite(df["Y"]) &
    np.isfinite(df["dx"]) &
    np.isfinite(df["dy"]) &
    np.isfinite(df["ZNCC"])
].copy()

df = df[
    df["ZNCC"] >= ZNCC_THRESHOLD
].copy()

print(
    f"Points after ZNCC >= {ZNCC_THRESHOLD}:",
    len(df)
)

if len(df) < NEIGHBORS:
    raise RuntimeError(
        "Not enough reliable DIC points."
    )


# ============================================================
# REMOVE DUPLICATE COORDINATES
# ============================================================

df = (
    df.groupby(["X", "Y"], as_index=False)
      .agg({
          "dx": "median",
          "dy": "median",
          "ZNCC": "median"
      })
)

print("Unique DIC points:", len(df))


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = cv2.imread(
    IMAGE_FILE,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        IMAGE_FILE
    )

height, width = image.shape

print(
    f"Image size: {width} x {height}"
)

print(
    f"DIC X: {df.X.min()} → {df.X.max()}"
)

print(
    f"DIC Y: {df.Y.min()} → {df.Y.max()}"
)


# ============================================================
# DIC ARRAYS
# ============================================================

coordinates = df[
    ["X", "Y"]
].to_numpy(dtype=float)

u = df["dx"].to_numpy(dtype=float)

v = df["dy"].to_numpy(dtype=float)

tree = cKDTree(coordinates)


# ============================================================
# LOCAL AFFINE STRAIN
# ============================================================

print("\nCalculating local affine deformation...")

results = []

total = len(coordinates)

for i, point in enumerate(coordinates):

    # --------------------------------------------------------
    # Find nearby DIC points
    # --------------------------------------------------------

    distances, indices = tree.query(
        point,
        k=min(
            NEIGHBORS,
            total
        )
    )

    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)

    # Remove points outside maximum neighborhood radius
    valid = distances <= MAX_RADIUS

    distances = distances[valid]
    indices = indices[valid]

    if len(indices) < 8:
        continue

    # --------------------------------------------------------
    # Local coordinates
    # --------------------------------------------------------

    local_xy = coordinates[indices]

    x_local = (
        local_xy[:, 0]
        - point[0]
    )

    y_local = (
        local_xy[:, 1]
        - point[1]
    )

    local_u = u[indices]

    local_v = v[indices]

    # --------------------------------------------------------
    # Weighted least-squares local affine model
    #
    # u = a + du/dx*x + du/dy*y
    # v = b + dv/dx*x + dv/dy*y
    # --------------------------------------------------------

    A = np.column_stack([
        np.ones(len(indices)),
        x_local,
        y_local
    ])

    # Give closer points greater influence.
    weights = 1.0 / (
        distances + 5.0
    )

    W = np.sqrt(weights)

    Aw = A * W[:, None]

    uw = local_u * W

    vw = local_v * W

    try:

        coeff_u, _, _, _ = np.linalg.lstsq(
            Aw,
            uw,
            rcond=None
        )

        coeff_v, _, _, _ = np.linalg.lstsq(
            Aw,
            vw,
            rcond=None
        )

    except np.linalg.LinAlgError:
        continue

    du_dx = coeff_u[1]
    du_dy = coeff_u[2]

    dv_dx = coeff_v[1]
    dv_dy = coeff_v[2]

    # --------------------------------------------------------
    # Small strain tensor
    # --------------------------------------------------------

    exx = du_dx

    eyy = dv_dy

    exy = 0.5 * (
        du_dy + dv_dx
    )

    # --------------------------------------------------------
    # Principal strains
    # --------------------------------------------------------

    mean_strain = 0.5 * (
        exx + eyy
    )

    radius = np.sqrt(
        (
            0.5 * (exx - eyy)
        ) ** 2
        +
        exy ** 2
    )

    principal_1 = (
        mean_strain + radius
    )

    principal_2 = (
        mean_strain - radius
    )

    results.append({
        "X": point[0],
        "Y": point[1],
        "Exx": exx,
        "Eyy": eyy,
        "Exy": exy,
        "Principal_Strain": principal_1,
        "Principal_Strain_2": principal_2,
        "ZNCC": df.iloc[i]["ZNCC"],
        "Neighbors_Used": len(indices),
        "Neighborhood_Radius": float(
            np.max(distances)
        )
    })

    if (i + 1) % 1000 == 0:
        print(
            f"Processed {i + 1}/{total}"
        )


# ============================================================
# RESULT DATAFRAME
# ============================================================

strain_df = pd.DataFrame(results)

if strain_df.empty:
    raise RuntimeError(
        "Local affine strain calculation "
        "produced no valid results."
    )

print(
    "\nValid local strain points:",
    len(strain_df)
)


# ============================================================
# SAVE POINT-LEVEL STRAIN
# ============================================================

strain_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_CSV
)


# ============================================================
# STRAIN STATISTICS
# ============================================================

strain_values = strain_df[
    "Principal_Strain"
].to_numpy()

strain_values = strain_values[
    np.isfinite(strain_values)
]

print("\n==============================================")
print("LOCAL AFFINE STRAIN STATISTICS")
print("==============================================")

print(
    f"Minimum : {np.min(strain_values):.8f}"
)

print(
    f"Maximum : {np.max(strain_values):.8f}"
)

print(
    f"Mean    : {np.mean(strain_values):.8f}"
)

print(
    f"P95     : {np.percentile(strain_values, 95):.8f}"
)

print(
    f"P99     : {np.percentile(strain_values, 99):.8f}"
)


# ============================================================
# VISUALIZATION GRID
# ============================================================

print("\nCreating visualization grid...")

x_values = np.arange(
    int(strain_df.X.min()),
    min(
        width,
        int(strain_df.X.max()) + 1
    ),
    GRID_STEP
)

y_values = np.arange(
    int(strain_df.Y.min()),
    min(
        height,
        int(strain_df.Y.max()) + 1
    ),
    GRID_STEP
)

grid_x, grid_y = np.meshgrid(
    x_values,
    y_values
)


# ============================================================
# INTERPOLATE LOCAL STRAIN
# ============================================================

points = strain_df[
    ["X", "Y"]
].to_numpy()

values = strain_df[
    "Principal_Strain"
].to_numpy()

print("Interpolating local strain...")

strain_grid = griddata(
    points,
    values,
    (grid_x, grid_y),
    method="linear"
)


# ============================================================
# LIGHT VISUAL SMOOTHING
# ============================================================

# Only for visualization.
# The saved point-level strain remains unsmoothed.

valid_mask = np.isfinite(
    strain_grid
)

filled = strain_grid.copy()

median_value = np.nanmedian(
    filled
)

filled[~valid_mask] = median_value

smooth_grid = gaussian_filter(
    filled,
    sigma=VISUAL_SMOOTHING
)

smooth_grid[~valid_mask] = np.nan


# ============================================================
# ROBUST COLOR LIMIT
# ============================================================

valid = smooth_grid[
    np.isfinite(smooth_grid)
]

vmin = np.percentile(
    valid,
    5
)

vmax = np.percentile(
    valid,
    99
)

if vmax <= vmin:
    vmax = np.max(valid)


# ============================================================
# CREATE HEATMAP
# ============================================================

print("\nGenerating heatmap...")

plt.figure(
    figsize=(17, 8)
)

plt.imshow(
    image,
    cmap="gray",
    origin="upper"
)

heatmap = np.ma.masked_invalid(
    smooth_grid
)

plt.imshow(
    heatmap,
    cmap="turbo",
    alpha=0.68,
    origin="upper",
    extent=[
        x_values.min(),
        x_values.max(),
        y_values.max(),
        y_values.min()
    ],
    vmin=vmin,
    vmax=vmax
)

cbar = plt.colorbar()

cbar.set_label(
    "Principal strain (DIC)"
)

plt.title(
    "Local Affine DIC Strain Heatmap\n"
    "Frame 1274 → 1374"
)

plt.xlabel(
    "X (pixels)"
)

plt.ylabel(
    "Y (pixels)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_HEATMAP,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("STAGE 26 COMPLETE")
print("==============================================")

print("\nHeatmap:")
print(OUTPUT_HEATMAP)

print("\nStrain CSV:")
print(OUTPUT_CSV)