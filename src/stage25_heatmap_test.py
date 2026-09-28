import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


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
    "Stage25_Heatmap_Test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "heatmap_1274_1374.png"
)


# ============================================================
# LOAD DIC DATA
# ============================================================

print("\nLoading DIC data...")

df = pd.read_csv(DIC_FILE)

required = ["X", "Y", "dx", "dy", "ZNCC"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print(f"DIC points: {len(df)}")


# ============================================================
# QUALITY FILTER
# ============================================================

# Keep high-quality correlation points
df = df[df["ZNCC"] >= 0.90].copy()

print(f"Points after ZNCC filtering: {len(df)}")


# ============================================================
# DISPLACEMENT MAGNITUDE
# ============================================================

df["magnitude"] = np.sqrt(
    df["dx"] ** 2 +
    df["dy"] ** 2
)


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_FILE}"
    )

height, width = image.shape

print(f"Image size: {width} x {height}")


# ============================================================
# CREATE REGULAR GRID
# ============================================================

print("\nCreating spatial grid...")

grid_x, grid_y = np.meshgrid(
    np.arange(0, width, 5),
    np.arange(0, height, 5)
)


# ============================================================
# INTERPOLATE DISPLACEMENT
# ============================================================

print("Interpolating displacement field...")

points = np.column_stack(
    (df["X"].values, df["Y"].values)
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

mag_grid = np.sqrt(
    dx_grid ** 2 +
    dy_grid ** 2
)


# ============================================================
# CALCULATE SPATIAL GRADIENTS
# ============================================================

print("Calculating spatial deformation field...")

# Grid spacing = 5 pixels
pixel_spacing = 5.0

du_dy, du_dx = np.gradient(
    dx_grid,
    pixel_spacing,
    pixel_spacing
)

dv_dy, dv_dx = np.gradient(
    dy_grid,
    pixel_spacing,
    pixel_spacing
)


# ============================================================
# PRINCIPAL STRAIN APPROXIMATION
# ============================================================

exx = du_dx
eyy = dv_dy

exy = 0.5 * (
    du_dy + dv_dx
)

mean_strain = 0.5 * (exx + eyy)

radius = np.sqrt(
    ((exx - eyy) * 0.5) ** 2 +
    exy ** 2
)

principal_strain = mean_strain + radius


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

principal_strain[
    ~np.isfinite(principal_strain)
] = np.nan


# ============================================================
# ROBUST HEATMAP RANGE
# ============================================================

valid = principal_strain[
    np.isfinite(principal_strain)
]

if len(valid) == 0:
    raise RuntimeError(
        "No valid strain values were generated."
    )

low = np.percentile(valid, 5)
high = np.percentile(valid, 99)

print("\nStrain statistics:")
print(f"Minimum : {np.min(valid):.6f}")
print(f"Maximum : {np.max(valid):.6f}")
print(f"Mean    : {np.mean(valid):.6f}")
print(f"P95     : {np.percentile(valid, 95):.6f}")
print(f"P99     : {np.percentile(valid, 99):.6f}")


# ============================================================
# PLOT
# ============================================================

print("\nGenerating heatmap...")

plt.figure(figsize=(16, 8))

plt.imshow(
    image,
    cmap="gray",
    origin="upper"
)

heatmap = np.ma.masked_invalid(principal_strain)

plt.imshow(
    heatmap,
    cmap="turbo",
    alpha=0.65,
    origin="upper",
    vmin=low,
    vmax=high
)

plt.colorbar(
    label="Principal strain (DIC)"
)

plt.title(
    "DIC Spatial Deformation Heatmap\n"
    "Frame 1274 → 1374"
)

plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n========================================")
print("HEATMAP GENERATED SUCCESSFULLY")
print("========================================")
print(OUTPUT_FILE)