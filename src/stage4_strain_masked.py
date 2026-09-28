import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DIC_FOLDER = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage3_DIC_Test"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage4_Strain_Test"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# INPUT DIC RESULT
# ==========================================

CSV_NAME = "dic_6500_6600.csv"

INPUT_CSV = os.path.join(
    DIC_FOLDER,
    CSV_NAME
)


# ==========================================
# FIXTURE MASK
# ==========================================
# Approximate black fixture region.
#
# IMPORTANT:
# This is only for this TEST.
# We will make the mask dynamic later.

FIXTURE_X_MIN = 930
FIXTURE_X_MAX = 1060

FIXTURE_Y_MIN = 490
FIXTURE_Y_MAX = 638


# ==========================================
# LOAD DIC DATA
# ==========================================

print("\n==========================================")
print("      MASKED DIC STRAIN TEST")
print("==========================================")

print("\nReading:")
print(INPUT_CSV)

if not os.path.exists(INPUT_CSV):

    print("\nERROR: CSV not found.")
    print(INPUT_CSV)
    exit()


df = pd.read_csv(INPUT_CSV)

print("\nOriginal DIC points:", len(df))


# ==========================================
# REMOVE FIXTURE POINTS
# ==========================================

fixture_mask = (
    (df["X"] >= FIXTURE_X_MIN) &
    (df["X"] <= FIXTURE_X_MAX) &
    (df["Y"] >= FIXTURE_Y_MIN) &
    (df["Y"] <= FIXTURE_Y_MAX)
)

removed_points = fixture_mask.sum()

df = df.loc[~fixture_mask].copy()

print(
    "Fixture points removed:",
    removed_points
)

print(
    "Remaining DIC points:",
    len(df)
)


# ==========================================
# EXTRACT DISPLACEMENT
# ==========================================

x = df["X"].values
y = df["Y"].values

u = df["dx"].values
v = df["dy"].values


# ==========================================
# CREATE REGULAR GRID
# ==========================================

x_grid = np.linspace(
    x.min(),
    x.max(),
    len(np.unique(x))
)

y_grid = np.linspace(
    y.min(),
    y.max(),
    len(np.unique(y))
)

X, Y = np.meshgrid(
    x_grid,
    y_grid
)


# ==========================================
# INTERPOLATE U
# ==========================================

print("\nInterpolating displacement field...")

U = griddata(
    (x, y),
    u,
    (X, Y),
    method="linear"
)

U_nearest = griddata(
    (x, y),
    u,
    (X, Y),
    method="nearest"
)

U = np.where(
    np.isnan(U),
    U_nearest,
    U
)


# ==========================================
# INTERPOLATE V
# ==========================================

V = griddata(
    (x, y),
    v,
    (X, Y),
    method="linear"
)

V_nearest = griddata(
    (x, y),
    v,
    (X, Y),
    method="nearest"
)

V = np.where(
    np.isnan(V),
    V_nearest,
    V
)


# ==========================================
# SMOOTH DISPLACEMENT FIELD
# ==========================================

U = gaussian_filter(
    U,
    sigma=2
)

V = gaussian_filter(
    V,
    sigma=2
)


# ==========================================
# CALCULATE SPATIAL GRADIENTS
# ==========================================

dx_spacing = (
    x_grid[1] - x_grid[0]
)

dy_spacing = (
    y_grid[1] - y_grid[0]
)


du_dy, du_dx = np.gradient(
    U,
    dy_spacing,
    dx_spacing
)

dv_dy, dv_dx = np.gradient(
    V,
    dy_spacing,
    dx_spacing
)


# ==========================================
# STRAIN COMPONENTS
# ==========================================

epsilon_xx = du_dx

epsilon_yy = dv_dy

epsilon_xy = 0.5 * (
    du_dy + dv_dx
)


# ==========================================
# PRINCIPAL STRAINS
# ==========================================

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

principal_1 = (
    mean_strain + radius
)

principal_2 = (
    mean_strain - radius
)


max_principal_strain = np.maximum(
    np.abs(principal_1),
    np.abs(principal_2)
)


# ==========================================
# MASK FIXTURE IN GRID
# ==========================================

grid_fixture_mask = (
    (X >= FIXTURE_X_MIN) &
    (X <= FIXTURE_X_MAX) &
    (Y >= FIXTURE_Y_MIN) &
    (Y <= FIXTURE_Y_MAX)
)

max_principal_strain[
    grid_fixture_mask
] = np.nan


# ==========================================
# SAVE STRAIN DATA
# ==========================================

strain_df = pd.DataFrame({

    "X": X.ravel(),

    "Y": Y.ravel(),

    "epsilon_xx":
        epsilon_xx.ravel(),

    "epsilon_yy":
        epsilon_yy.ravel(),

    "epsilon_xy":
        epsilon_xy.ravel(),

    "principal_strain_1":
        principal_1.ravel(),

    "principal_strain_2":
        principal_2.ravel(),

    "max_principal_strain":
        max_principal_strain.ravel()
})

strain_csv = os.path.join(
    OUTPUT_FOLDER,
    "strain_6500_6600_masked.csv"
)

strain_df.to_csv(
    strain_csv,
    index=False
)


# ==========================================
# STATISTICS
# ==========================================

valid_strain = (
    max_principal_strain[
        ~np.isnan(max_principal_strain)
    ]
)

print("\n==========================================")
print("        STRAIN STATISTICS")
print("==========================================")

print(
    "\nMaximum principal strain:",
    np.max(valid_strain)
)

print(
    "Mean principal strain:",
    np.mean(valid_strain)
)

print(
    "95th percentile:",
    np.percentile(
        valid_strain,
        95
    )
)

print(
    "99th percentile:",
    np.percentile(
        valid_strain,
        99
    )
)


# ==========================================
# FIND MAXIMUM LOCATION
# ==========================================

max_index = np.nanargmax(
    max_principal_strain
)

max_row, max_col = np.unravel_index(
    max_index,
    max_principal_strain.shape
)

max_x = X[
    max_row,
    max_col
]

max_y = Y[
    max_row,
    max_col
]

max_value = max_principal_strain[
    max_row,
    max_col
]


print("\n==========================================")
print("     MAXIMUM STRAIN LOCATION")
print("==========================================")

print(
    "X:",
    round(float(max_x), 2)
)

print(
    "Y:",
    round(float(max_y), 2)
)

print(
    "Maximum strain:",
    max_value
)


# ==========================================
# HEATMAP
# ==========================================

print("\nGenerating heatmap...")


plt.figure(
    figsize=(15, 6)
)


# Robust scale
vmax = np.percentile(
    valid_strain,
    99
)

if vmax <= 0:

    vmax = np.max(
        valid_strain
    )


plt.imshow(

    max_principal_strain,

    cmap="jet",

    origin="upper",

    interpolation="bicubic",

    aspect="auto",

    extent=[
        x.min(),
        x.max(),
        y.max(),
        y.min()
    ],

    vmin=0,

    vmax=vmax
)


plt.colorbar(
    label="Maximum Principal Strain"
)


# Mark maximum point

plt.scatter(

    max_x,

    max_y,

    s=140,

    facecolors="none",

    edgecolors="white",

    linewidths=2
)


plt.text(

    max_x + 15,

    max_y,

    f"Maximum\n({max_x:.0f}, {max_y:.0f})",

    color="white",

    fontsize=10,

    fontweight="bold"
)


plt.title(

    "DIC Maximum Principal Strain\n"
    "Frame 6500 → 6600",

    fontsize=16,

    fontweight="bold"
)


plt.xlabel(
    "X Position (pixels)"
)

plt.ylabel(
    "Y Position (pixels)"
)


plt.tight_layout()


heatmap_path = os.path.join(

    OUTPUT_FOLDER,

    "strain_heatmap_6500_6600_masked.png"
)


plt.savefig(

    heatmap_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ==========================================
# FINISH
# ==========================================

print("\n==========================================")
print("       MASKED STRAIN TEST COMPLETE")
print("==========================================")

print("\nStrain CSV:")
print(strain_csv)

print("\nHeatmap:")
print(heatmap_path)