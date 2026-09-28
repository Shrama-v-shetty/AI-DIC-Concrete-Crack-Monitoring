import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "Stage25B_Smoothed_Heatmap"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "smoothed_strain_heatmap_1274_1374.png"
)

STRAIN_CSV = os.path.join(
    OUTPUT_DIR,
    "smoothed_strain_1274_1374.csv"
)


# ============================================================
# LOAD DIC
# ============================================================

print("\n========================================")
print("STAGE 25B - SMOOTHED DIC HEATMAP")
print("========================================")

print("\nLoading DIC data...")

df = pd.read_csv(DIC_FILE)

required = ["X", "Y", "dx", "dy", "ZNCC"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print("DIC points:", len(df))

print(
    "DIC coordinate range:",
    f"X={df.X.min()} to {df.X.max()},",
    f"Y={df.Y.min()} to {df.Y.max()}"
)


# ============================================================
# QUALITY FILTER
# ============================================================

# Strong correlation points only
df = df[df["ZNCC"] >= 0.95].copy()

print("Points after ZNCC >= 0.95:", len(df))

if len(df) < 100:
    raise RuntimeError(
        "Too few reliable DIC points after ZNCC filtering."
    )


# ============================================================
# LOAD TARGET IMAGE
# ============================================================

print("\nLoading target image...")

image = cv2.imread(
    IMAGE_FILE,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"\nCould not load image:\n{IMAGE_FILE}"
    )

height, width = image.shape

print("Image width :", width)
print("Image height:", height)


# ============================================================
# CHECK COORDINATE COMPATIBILITY
# ============================================================

x_min = df["X"].min()
x_max = df["X"].max()

y_min = df["Y"].min()
y_max = df["Y"].max()

print("\nCoordinate compatibility check:")

print(
    f"Image X range: 0 to {width - 1}"
)

print(
    f"DIC X range:   {x_min} to {x_max}"
)

print(
    f"Image Y range: 0 to {height - 1}"
)

print(
    f"DIC Y range:   {y_min} to {y_max}"
)


# ============================================================
# GRID
# ============================================================

# Use the actual DIC coordinate region.
# 5 pixel spacing gives a sufficiently smooth field.

grid_step = 5

grid_x, grid_y = np.meshgrid(
    np.arange(
        max(0, int(x_min)),
        min(width, int(x_max) + 1),
        grid_step
    ),
    np.arange(
        max(0, int(y_min)),
        min(height, int(y_max) + 1),
        grid_step
    )
)


# ============================================================
# INTERPOLATE DX AND DY
# ============================================================

print("\nInterpolating DIC displacement field...")

points = np.column_stack(
    (
        df["X"].values,
        df["Y"].values
    )
)

dx_grid = griddata(
    points,
    df["dx"].values,
    (grid_x, grid_y),
    method="linear"
)

dy_grid = griddata(
    points,
    df["dy"].values,
    (grid_x, grid_y),
    method="linear"
)


# ============================================================
# VALID MASK
# ============================================================

valid_mask = (
    np.isfinite(dx_grid)
    &
    np.isfinite(dy_grid)
)


# ============================================================
# FILL SMALL INTERNAL GAPS
# ============================================================

print("Preparing displacement field...")

def fill_nan(field):
    result = field.copy()

    valid = np.isfinite(result)

    if not np.any(valid):
        return result

    mean_value = np.nanmedian(result)

    result[~valid] = mean_value

    return result


dx_filled = fill_nan(dx_grid)
dy_filled = fill_nan(dy_grid)


# ============================================================
# SMOOTH DISPLACEMENT
# ============================================================

print("Smoothing displacement field...")

# DIC points are approximately 10 pixels apart.
# Sigma=2 means moderate spatial smoothing.

sigma = 2.0

dx_smooth = gaussian_filter(
    dx_filled,
    sigma=sigma
)

dy_smooth = gaussian_filter(
    dy_filled,
    sigma=sigma
)


# ============================================================
# CALCULATE DISPLACEMENT GRADIENTS
# ============================================================

print("Calculating spatial deformation...")

# Grid spacing is 5 pixels.
spacing = float(grid_step)

du_dy, du_dx = np.gradient(
    dx_smooth,
    spacing,
    spacing
)

dv_dy, dv_dx = np.gradient(
    dy_smooth,
    spacing,
    spacing
)


# ============================================================
# SMALL-STRAIN TENSOR
# ============================================================

exx = du_dx

eyy = dv_dy

exy = 0.5 * (
    du_dy + dv_dx
)


# ============================================================
# PRINCIPAL STRAIN
# ============================================================

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

principal_strain = (
    mean_strain + radius
)


# ============================================================
# RESTORE INVALID AREAS
# ============================================================

principal_strain[
    ~valid_mask
] = np.nan


# ============================================================
# REMOVE EXTREME SINGLE-POINT OUTLIERS
# ============================================================

valid_values = principal_strain[
    np.isfinite(principal_strain)
]

if len(valid_values) == 0:
    raise RuntimeError(
        "No valid strain values were produced."
    )


# Robust visualization limits
p2 = np.percentile(valid_values, 2)
p98 = np.percentile(valid_values, 98)

if p98 <= p2:
    p98 = np.max(valid_values)

print("\n========================================")
print("STRAIN STATISTICS")
print("========================================")

print(f"Minimum : {np.min(valid_values):.8f}")
print(f"Maximum : {np.max(valid_values):.8f}")
print(f"Mean    : {np.mean(valid_values):.8f}")
print(f"P95     : {np.percentile(valid_values, 95):.8f}")
print(f"P98     : {p98:.8f}")
print(f"P99     : {np.percentile(valid_values, 99):.8f}")


# ============================================================
# SAVE STRAIN FIELD
# ============================================================

strain_output = pd.DataFrame({
    "X": grid_x[valid_mask],
    "Y": grid_y[valid_mask],
    "Principal_Strain": principal_strain[valid_mask],
    "Smoothed_dx": dx_smooth[valid_mask],
    "Smoothed_dy": dy_smooth[valid_mask]
})

strain_output.to_csv(
    STRAIN_CSV,
    index=False
)

print("\nSaved strain data:")
print(STRAIN_CSV)


# ============================================================
# PREPARE IMAGE
# ============================================================

# Convert grayscale to RGB for overlay
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2RGB
)


# ============================================================
# HEATMAP
# ============================================================

print("\nGenerating heatmap...")

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    image_rgb,
    origin="upper"
)

heatmap = np.ma.masked_invalid(
    principal_strain
)

plt.imshow(
    heatmap,
    origin="upper",
    cmap="turbo",
    alpha=0.70,
    vmin=p2,
    vmax=p98,
    extent=[
        grid_x.min(),
        grid_x.max(),
        grid_y.max(),
        grid_y.min()
    ]
)

cbar = plt.colorbar()

cbar.set_label(
    "Principal strain (DIC)"
)

plt.title(
    "Smoothed DIC Spatial Strain Heatmap\n"
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
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("HEATMAP GENERATED SUCCESSFULLY")
print("========================================")

print("\nHeatmap:")
print(OUTPUT_FILE)

print("\nStrain CSV:")
print(STRAIN_CSV)

print("\nStage 25B complete.")