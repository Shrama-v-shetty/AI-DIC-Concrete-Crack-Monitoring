import pandas as pd
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def calculate_strain(input_csv, output_csv):

    # ==========================================
    # Load displacement data
    # ==========================================

    df = pd.read_csv(input_csv)

    # Grid coordinates
    x_values = np.sort(df["X"].unique())
    y_values = np.sort(df["Y"].unique())

    rows = len(y_values)
    cols = len(x_values)

    # Grid spacing
    step = x_values[1] - x_values[0]

    print("Grid spacing :", step)

    # ==========================================
    # Create displacement grids
    # ==========================================

    dx_grid = np.full((rows, cols), np.nan)
    dy_grid = np.full((rows, cols), np.nan)

    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}

    # Fill displacement grids

    for _, row in df.iterrows():

        xi = x_index[row["X"]]
        yi = y_index[row["Y"]]

        dx_grid[yi, xi] = row["dx"]
        dy_grid[yi, xi] = row["dy"]

    # ==========================================
    # Replace NaN values
    # ==========================================

    dx_grid = np.nan_to_num(dx_grid)
    dy_grid = np.nan_to_num(dy_grid)

    # ==========================================
    # Remove noisy vectors
    # ==========================================

    dx_grid = median_filter(dx_grid, size=3)
    dy_grid = median_filter(dy_grid, size=3)

    # ==========================================
    # Smooth displacement field
    # ==========================================

    dx_grid = gaussian_filter(dx_grid, sigma=1)
    dy_grid = gaussian_filter(dy_grid, sigma=1)

    # ==========================================
    # Calculate strain using numerical gradients
    # ==========================================

    eps_xx = np.gradient(dx_grid, step, axis=1)

    eps_yy = np.gradient(dy_grid, step, axis=0)

    du_dy = np.gradient(dx_grid, step, axis=0)

    dv_dx = np.gradient(dy_grid, step, axis=1)

    gamma_xy = du_dy + dv_dx

    # ==========================================
    # Print Statistics
    # ==========================================

    print("\n========== STRAIN STATISTICS ==========")

    print("\nεxx")
    print("Minimum :", np.min(eps_xx))
    print("Maximum :", np.max(eps_xx))
    print("Average :", np.mean(eps_xx))

    print("\nεyy")
    print("Minimum :", np.min(eps_yy))
    print("Maximum :", np.max(eps_yy))
    print("Average :", np.mean(eps_yy))

    print("\nγxy")
    print("Minimum :", np.min(gamma_xy))
    print("Maximum :", np.max(gamma_xy))
    print("Average :", np.mean(gamma_xy))

    # ==========================================
    # Save strain CSV
    # ==========================================

    strain_points = []

    for i, y in enumerate(y_values):

        for j, x in enumerate(x_values):

            strain_points.append([

                x,
                y,

                dx_grid[i, j],
                dy_grid[i, j],

                eps_xx[i, j],
                eps_yy[i, j],
                gamma_xy[i, j]

            ])

    strain_df = pd.DataFrame(

        strain_points,

        columns=[

            "X",
            "Y",

            "dx",
            "dy",

            "eps_xx",
            "eps_yy",
            "gamma_xy"

        ]

    )

    strain_df.to_csv(output_csv, index=False)

    print("\nSaved :", output_csv)


# ==========================================
# Run independently
# ==========================================

if __name__ == "__main__":

    calculate_strain(
        "../output/displacement.csv",
        "../output/strain.csv"
    )