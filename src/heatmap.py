import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# ==========================================
# Base Directory
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage3_DIC_Test"
)

# ==========================================
# Heatmap Function
# ==========================================

def generate_heatmap(input_csv, output_image):

    # --------------------------------------
    # Load DIC data
    # --------------------------------------

    df = pd.read_csv(input_csv)

    print("DIC points:", len(df))
    print("Columns:", list(df.columns))

    x_values = np.sort(df["X"].unique())
    y_values = np.sort(df["Y"].unique())

    rows = len(y_values)
    cols = len(x_values)

    heatmap = np.full((rows, cols), np.nan)

    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}

    # --------------------------------------
    # Fill displacement values
    # --------------------------------------

    for _, row in df.iterrows():

        xi = x_index[row["X"]]
        yi = y_index[row["Y"]]

        heatmap[yi, xi] = row["Displacement"]

    # --------------------------------------
    # Replace missing values
    # --------------------------------------

    heatmap = np.nan_to_num(heatmap)

    # --------------------------------------
    # Smooth Heatmap
    # --------------------------------------

    heatmap = gaussian_filter(
        heatmap,
        sigma=1
    )

    # --------------------------------------
    # Automatic Color Scale
    # --------------------------------------

    vmax = np.percentile(
        heatmap,
        99
    )

    # --------------------------------------
    # Plot Heatmap
    # --------------------------------------

    plt.figure(figsize=(14, 5))

    plt.imshow(
        heatmap,
        cmap="turbo",
        origin="lower",
        interpolation="bicubic",
        aspect="auto",
        vmin=0,
        vmax=vmax,
        extent=[
            min(x_values),
            max(x_values),
            min(y_values),
            max(y_values)
        ]
    )

    plt.colorbar(
        label="DIC Displacement (pixels)"
    )

    plt.title(
        "DIC Displacement Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "X Position (pixels)",
        fontsize=12
    )

    plt.ylabel(
        "Y Position (pixels)",
        fontsize=12
    )

    plt.tight_layout()

    plt.savefig(
        output_image,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n======================================")
    print("Heatmap Generated Successfully")
    print("Saved to :", os.path.abspath(output_image))
    print("======================================")


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":

    input_csv = os.path.join(
        OUTPUT_DIR,
        "dic_2525_2526.csv"
    )

    output_image = os.path.join(
        OUTPUT_DIR,
        "heatmap_2525_2526.png"
    )

    generate_heatmap(
        input_csv,
        output_image
    )