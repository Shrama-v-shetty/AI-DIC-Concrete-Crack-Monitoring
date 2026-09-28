import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test\dic_1000_1274.csv"
OUTPUT_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage4_Strain_Test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# APPROXIMATE FIXTURE REGION
# Same region used in the previous strain test
# ============================================================

FIXTURE_X_MIN = 930
FIXTURE_X_MAX = 1060

FIXTURE_Y_MIN = 490
FIXTURE_Y_MAX = 638


# ============================================================
# READ DIC DATA
# ============================================================

print("==========================================")
print("      MASKED DIC STRAIN TEST")
print("      FRAME 1000 -> 1274")
print("==========================================")

print("\nReading:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print("\nOriginal DIC points:", len(df))


# ============================================================
# REMOVE FIXTURE POINTS
# ============================================================

fixture_mask = (
    (df["X"] >= FIXTURE_X_MIN) &
    (df["X"] <= FIXTURE_X_MAX) &
    (df["Y"] >= FIXTURE_Y_MIN) &
    (df["Y"] <= FIXTURE_Y_MAX)
)

df = df[~fixture_mask].copy()

print("Fixture points removed:", fixture_mask.sum())
print("Remaining DIC points:", len(df))


# ============================================================
# EXTRACT DIC POINTS
# ============================================================

x = df["X"].values
y = df["Y"].values

u = df["dx"].values
v = df["dy"].values


# ============================================================
# CREATE REGULAR GRID
# ============================================================

print("\nInterpolating displacement field...")

grid_x = np.linspace(
    x.min(),
    x.max(),
    350
)

grid_y = np.linspace(
    y.min(),
    y.max(),
    130
)

X, Y = np.meshgrid(
    grid_x,
    grid_y
)


# ============================================================
# INTERPOLATE U AND V
# ============================================================

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


# ============================================================
# FILL MISSING VALUES
# ============================================================

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


# ============================================================
# SMOOTH DISPLACEMENT FIELD
# ============================================================

U = gaussian_filter(U, sigma=2)
V = gaussian_filter(V, sigma=2)


# ============================================================
# CALCULATE DISPLACEMENT GRADIENTS
# ============================================================

du_dy, du_dx = np.gradient(
    U,
    grid_y,
    grid_x
)

dv_dy, dv_dx = np.gradient(
    V,
    grid_y,
    grid_x
)


# ============================================================
# STRAIN COMPONENTS
# ============================================================

epsilon_xx = du_dx

epsilon_yy = dv_dy

epsilon_xy = 0.5 * (
    du_dy + dv_dx
)


# ============================================================
# PRINCIPAL STRAIN
# ============================================================

mean_strain = (
    epsilon_xx + epsilon_yy
) / 2

radius = np.sqrt(
    ((epsilon_xx - epsilon_yy) / 2) ** 2
    + epsilon_xy ** 2
)

principal_1 = mean_strain + radius

principal_2 = mean_strain - radius

max_principal_strain = principal_1


# ============================================================
# MASK FIXTURE REGION ON GRID
# ============================================================

fixture_grid_mask = (
    (X >= FIXTURE_X_MIN) &
    (X <= FIXTURE_X_MAX) &
    (Y >= FIXTURE_Y_MIN) &
    (Y <= FIXTURE_Y_MAX)
)

max_principal_strain = np.where(
    fixture_grid_mask,
    np.nan,
    max_principal_strain
)


# ============================================================
# STATISTICS
# ============================================================

valid_strain = max_principal_strain[
    ~np.isnan(max_principal_strain)
]

print("\n==========================================")
print("        STRAIN STATISTICS")
print("==========================================")

print(
    "Maximum principal strain:",
    np.nanmax(valid_strain)
)

print(
    "Mean principal strain:",
    np.nanmean(valid_strain)
)

print(
    "95th percentile:",
    np.percentile(valid_strain, 95)
)

print(
    "99th percentile:",
    np.percentile(valid_strain, 99)
)


# ============================================================
# MAXIMUM STRAIN LOCATION
# ============================================================

max_index = np.nanargmax(
    max_principal_strain
)

max_y_index, max_x_index = np.unravel_index(
    max_index,
    max_principal_strain.shape
)

max_x = X[max_y_index, max_x_index]
max_y = Y[max_y_index, max_x_index]

max_value = max_principal_strain[
    max_y_index,
    max_x_index
]


print("\n==========================================")
print("     MAXIMUM STRAIN LOCATION")
print("==========================================")

print("X:", max_x)
print("Y:", max_y)

print(
    "Maximum strain:",
    max_value
)


# ============================================================
# SAVE STRAIN DATA
# ============================================================

strain_df = pd.DataFrame({
    "X": X.ravel(),
    "Y": Y.ravel(),
    "U": U.ravel(),
    "V": V.ravel(),
    "epsilon_xx": epsilon_xx.ravel(),
    "epsilon_yy": epsilon_yy.ravel(),
    "epsilon_xy": epsilon_xy.ravel(),
    "principal_strain": max_principal_strain.ravel()
})

csv_output = os.path.join(
    OUTPUT_DIR,
    "strain_1000_1274_masked.csv"
)
strain_df.to_csv(
    csv_output,
    index=False
)


# ============================================================
# HEATMAP
# ============================================================

print("\nGenerating heatmap...")

plt.figure(
    figsize=(14, 5)
)

plt.imshow(
    max_principal_strain,
    cmap="jet",
    origin="upper",
    interpolation="bicubic",
    aspect="auto",
    extent=[
        grid_x.min(),
        grid_x.max(),
        grid_y.max(),
        grid_y.min()
    ]
)

plt.colorbar(
    label="Maximum Principal Strain"
)

plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")

plt.title(
    "DIC Maximum Principal Strain\n"
    "Frame 1000 → 1274"
)

plt.tight_layout()


heatmap_output = os.path.join(
    OUTPUT_DIR,
    "strain_heatmap_1000_1274_masked.png"
)

plt.savefig(
    heatmap_output,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("       MASKED STRAIN TEST COMPLETE")
print("==========================================")

print("\nStrain CSV:")
print(csv_output)

print("\nHeatmap:")
print(heatmap_output)