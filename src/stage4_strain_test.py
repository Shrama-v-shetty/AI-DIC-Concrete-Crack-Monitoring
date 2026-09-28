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
# TEST PAIR
# ==========================================

CSV_NAME = "dic_6599_6600.csv"

INPUT_CSV = os.path.join(
    DIC_FOLDER,
    CSV_NAME
)


# ==========================================
# STRAIN CALCULATION
# ==========================================

def calculate_strain(csv_path):

    print("\n==========================================")
    print("          DIC STRAIN CALCULATION")
    print("==========================================")

    print("\nReading:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("\nDIC points:", len(df))

    # --------------------------------------
    # Original DIC coordinates
    # --------------------------------------

    x = df["X"].values
    y = df["Y"].values

    u = df["dx"].values
    v = df["dy"].values

    # --------------------------------------
    # Create regular grid
    # --------------------------------------

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

    print("\nInterpolating DIC displacement field...")

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

    # --------------------------------------
    # Fill remaining holes
    # --------------------------------------

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

    # --------------------------------------
    # Smooth displacement field
    # --------------------------------------

    U = gaussian_filter(
        U,
        sigma=1
    )

    V = gaussian_filter(
        V,
        sigma=1
    )

    # --------------------------------------
    # Spatial gradients
    # --------------------------------------

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

    # --------------------------------------
    # Strain components
    # --------------------------------------

    epsilon_xx = du_dx

    epsilon_yy = dv_dy

    epsilon_xy = 0.5 * (
        du_dy + dv_dx
    )

    # --------------------------------------
    # Maximum principal strain
    # --------------------------------------

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

    # --------------------------------------
    # Maximum absolute principal strain
    # --------------------------------------

    max_principal_strain = np.maximum(
        np.abs(principal_1),
        np.abs(principal_2)
    )

    # ======================================
    # SAVE STRAIN DATA
    # ======================================

    strain_df = pd.DataFrame({

        "X": X.ravel(),

        "Y": Y.ravel(),

        "epsilon_xx": epsilon_xx.ravel(),

        "epsilon_yy": epsilon_yy.ravel(),

        "epsilon_xy": epsilon_xy.ravel(),

        "principal_strain_1": principal_1.ravel(),

        "principal_strain_2": principal_2.ravel(),

        "max_principal_strain":
            max_principal_strain.ravel()
    })

    output_csv = os.path.join(
        OUTPUT_FOLDER,
        "strain_6599_6600.csv"
    )

    strain_df.to_csv(
        output_csv,
        index=False
    )

    print("\nStrain data saved:")
    print(output_csv)

    # ======================================
    # STATISTICS
    # ======================================

    print("\n==========================================")
    print("          STRAIN STATISTICS")
    print("==========================================")

    print(
        "\nMaximum |epsilon_xx|:",
        np.max(np.abs(epsilon_xx))
    )

    print(
        "Maximum |epsilon_yy|:",
        np.max(np.abs(epsilon_yy))
    )

    print(
        "Maximum |epsilon_xy|:",
        np.max(np.abs(epsilon_xy))
    )

    print(
        "Maximum principal strain:",
        np.max(max_principal_strain)
    )

    # ======================================
    # FIND MAXIMUM PRINCIPAL STRAIN POINT
    # ======================================

    max_index = np.unravel_index(
        np.argmax(max_principal_strain),
        max_principal_strain.shape
    )

    max_x = X[max_index]
    max_y = Y[max_index]

    max_value = max_principal_strain[
        max_index
    ]

    print("\n==========================================")
    print("     MAXIMUM PRINCIPAL STRAIN POINT")
    print("==========================================")

    print("X:", round(float(max_x), 2))
    print("Y:", round(float(max_y), 2))
    print("Value:", max_value)

    # ======================================
    # HEATMAP
    # ======================================

    print("\nGenerating strain heatmap...")

    plt.figure(
        figsize=(15, 6)
    )

    # Robust automatic scale
    vmax = np.percentile(
        max_principal_strain,
        99
    )

    if vmax <= 0:
        vmax = np.max(
            max_principal_strain
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

    plt.scatter(
        max_x,
        max_y,
        s=120,
        facecolors="none",
        edgecolors="white",
        linewidths=2
    )

    plt.text(
        max_x + 15,
        max_y,
        f"Max: {max_value:.4e}",
        color="white",
        fontsize=11,
        fontweight="bold"
    )

    plt.title(
        "DIC Maximum Principal Strain Field: 6599 → 6600",
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
        "strain_heatmap_6599_6600.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nHeatmap saved:")
    print(heatmap_path)

    print("\n==========================================")
    print("       STRAIN TEST COMPLETE")
    print("==========================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    if not os.path.exists(INPUT_CSV):

        print("\nERROR:")
        print("CSV file not found:")
        print(INPUT_CSV)

        exit()

    calculate_strain(
        INPUT_CSV
    )