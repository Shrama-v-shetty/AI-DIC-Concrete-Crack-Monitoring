import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\shrama\Desktop\Dam updated dataset"

DIC_DIR = os.path.join(
    ROOT,
    "Dataset",
    "Stage6C_Crack_Progression"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "Dataset",
    "BGSL_ROI"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "Dataset",
    "Stage8_Displacement_Validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INTERVALS
# ============================================================

INTERVALS = [
    (1274, 1500),
    (2000, 2500),
    (3000, 3432)
]


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

GRID_STEP = 10

SMOOTH_SIGMA = 1.0


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(frame):

    filename = (
        f"BGSL-2-{frame:04d}_0.tif"
    )

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# CREATE GRID
# ============================================================

def make_grid(
    x,
    y,
    values,
    width,
    height
):

    gx_values = np.arange(
        0,
        width,
        GRID_STEP
    )

    gy_values = np.arange(
        0,
        height,
        GRID_STEP
    )

    gx, gy = np.meshgrid(
        gx_values,
        gy_values
    )

    grid = griddata(
        (x, y),
        values,
        (gx, gy),
        method="linear"
    )

    missing = np.isnan(grid)

    if np.any(missing):

        grid[missing] = griddata(
            (x, y),
            values,
            (
                gx[missing],
                gy[missing]
            ),
            method="nearest"
        )

    grid = gaussian_filter(
        grid,
        sigma=SMOOTH_SIGMA
    )

    return gx, gy, grid


# ============================================================
# SAVE ONE MAP
# ============================================================

def save_map(
    image,
    gx,
    gy,
    grid,
    title,
    filename,
    cmap="viridis",
    symmetric=False
):

    plt.figure(
        figsize=(17, 7)
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    if symmetric:

        limit = np.percentile(
            np.abs(grid),
            99
        )

        limit = max(
            limit,
            1e-8
        )

        im = plt.pcolormesh(
            gx,
            gy,
            grid,
            cmap=cmap,
            shading="auto",
            alpha=0.75,
            vmin=-limit,
            vmax=limit
        )

    else:

        vmin = np.percentile(
            grid,
            1
        )

        vmax = np.percentile(
            grid,
            99
        )

        im = plt.pcolormesh(
            gx,
            gy,
            grid,
            cmap=cmap,
            shading="auto",
            alpha=0.75,
            vmin=vmin,
            vmax=vmax
        )

    plt.colorbar(
        im,
        label=title
    )

    plt.title(
        title
    )

    plt.xlabel(
        "X (pixels)"
    )

    plt.ylabel(
        "Y (pixels)"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        filename
    )


# ============================================================
# PROCESS
# ============================================================

print("==============================================")
print(" STAGE 8 - DIC DISPLACEMENT VALIDATION")
print("==============================================")


for start_frame, end_frame in INTERVALS:

    print("\n========================================")
    print(
        f"Processing {start_frame} -> {end_frame}"
    )
    print("========================================")

    dic_file = os.path.join(
        DIC_DIR,
        f"dic_{start_frame}_{end_frame}.csv"
    )

    image_file = find_image(
        end_frame
    )

    if not os.path.exists(dic_file):

        print(
            "DIC file missing:",
            dic_file
        )

        continue

    if image_file is None:

        print(
            "Image missing:",
            end_frame
        )

        continue

    # --------------------------------------------------------
    # Read DIC
    # --------------------------------------------------------

    df = pd.read_csv(
        dic_file
    )

    print(
        "Original points:",
        len(df)
    )

    # --------------------------------------------------------
    # ZNCC filtering
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ].copy()

    print(
        "After ZNCC filtering:",
        len(df)
    )

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    for column in [
        "X",
        "Y",
        "dx",
        "dy",
        "ZNCC"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    import cv2

    image = cv2.imread(
        image_file,
        cv2.IMREAD_GRAYSCALE
    )

    height, width = image.shape

    print(
        "Image:",
        width,
        "x",
        height
    )

    # --------------------------------------------------------
    # Extract DIC values
    # --------------------------------------------------------

    x = df["X"].values
    y = df["Y"].values

    dx = df["dx"].values
    dy = df["dy"].values

    displacement = np.sqrt(
        dx ** 2 +
        dy ** 2
    )

    zncc = df["ZNCC"].values

    # --------------------------------------------------------
    # Create grids
    # --------------------------------------------------------

    gx, gy, grid_dx = make_grid(
        x,
        y,
        dx,
        width,
        height
    )

    _, _, grid_dy = make_grid(
        x,
        y,
        dy,
        width,
        height
    )

    _, _, grid_disp = make_grid(
        x,
        y,
        displacement,
        width,
        height
    )

    _, _, grid_zncc = make_grid(
        x,
        y,
        zncc,
        width,
        height
    )

    interval = (
        f"{start_frame}_{end_frame}"
    )

    # --------------------------------------------------------
    # DX
    # --------------------------------------------------------

    save_map(
        image,
        gx,
        gy,
        grid_dx,
        f"DIC Horizontal Displacement (dx) — {interval}",
        os.path.join(
            OUTPUT_DIR,
            f"dx_{interval}.png"
        ),
        cmap="coolwarm",
        symmetric=True
    )

    # --------------------------------------------------------
    # DY
    # --------------------------------------------------------

    save_map(
        image,
        gx,
        gy,
        grid_dy,
        f"DIC Vertical Displacement (dy) — {interval}",
        os.path.join(
            OUTPUT_DIR,
            f"dy_{interval}.png"
        ),
        cmap="coolwarm",
        symmetric=True
    )

    # --------------------------------------------------------
    # MAGNITUDE
    # --------------------------------------------------------

    save_map(
        image,
        gx,
        gy,
        grid_disp,
        f"DIC Displacement Magnitude — {interval}",
        os.path.join(
            OUTPUT_DIR,
            f"magnitude_{interval}.png"
        ),
        cmap="viridis",
        symmetric=False
    )

    # --------------------------------------------------------
    # ZNCC
    # --------------------------------------------------------

    save_map(
        image,
        gx,
        gy,
        grid_zncc,
        f"DIC ZNCC Quality — {interval}",
        os.path.join(
            OUTPUT_DIR,
            f"zncc_{interval}.png"
        ),
        cmap="viridis",
        symmetric=False
    )

    # --------------------------------------------------------
    # Save raw validation data
    # --------------------------------------------------------

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"validation_{interval}.csv"
    )

    df[
        [
            "X",
            "Y",
            "dx",
            "dy",
            "ZNCC"
        ]
    ].assign(
        Displacement=displacement
    ).to_csv(
        output_csv,
        index=False
    )

    print(
        "Saved:",
        output_csv
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nDIC FIELD STATISTICS")

    print(
        f"Mean dx          : {np.mean(dx):.6f}"
    )

    print(
        f"Mean dy          : {np.mean(dy):.6f}"
    )

    print(
        f"Mean displacement: "
        f"{np.mean(displacement):.6f}"
    )

    print(
        f"95% displacement : "
        f"{np.percentile(displacement,95):.6f}"
    )

    print(
        f"99% displacement : "
        f"{np.percentile(displacement,99):.6f}"
    )

    print(
        f"Mean ZNCC        : "
        f"{np.mean(zncc):.6f}"
    )


print("\n==============================================")
print(" STAGE 8 COMPLETE")
print("==============================================")