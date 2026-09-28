import os
import glob
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap


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
    "Stage7_Local_Affine_Strain"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.985

# Radius around each DIC point used for local fitting
LOCAL_RADIUS = 60

# Minimum neighbouring DIC points
MIN_NEIGHBORS = 12

# Gaussian smoothing of final strain field
STRAIN_SMOOTH_SIGMA = 1.2

# Grid used only for visualization
GRID_STEP = 10


INTERVALS = [
    (1274, 1500),
    (1500, 1700),
    (1700, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3432),
]


# ============================================================
# LOADING REGION
#
# Used to prevent the loading/fixture area from dominating
# rigid-body estimation and final statistics.
# ============================================================

LOADING_X_MIN = 900
LOADING_X_MAX = 1080

LOADING_Y_MIN = 450
LOADING_Y_MAX = 638


# ============================================================
# ACTUAL FIXTURE POLYGON
#
# Used only to remove the black fixture itself.
# ============================================================

FIXTURE_POLYGON = np.array([
    [955, 638],
    [1020, 638],
    [1010, 545],
    [1007, 525],
    [1000, 513],
    [990, 507],
    [978, 506],
    [968, 512],
    [963, 523],
    [960, 540],
    [958, 560]
], dtype=np.int32)


# ============================================================
# HEATMAP COLOUR MAP
# ============================================================

DIC_CMAP = LinearSegmentedColormap.from_list(
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
# FRAME NUMBER
# ============================================================

def get_frame_number(filename):

    numbers = re.findall(
        r"\d+",
        os.path.basename(filename)
    )

    if len(numbers) < 2:
        return None

    return int(numbers[-2])


# ============================================================
# IMAGE MAP
# ============================================================

def load_image_map():

    files = glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.tif"
        )
    )

    image_map = {}

    for file in files:

        frame = get_frame_number(file)

        if frame is not None:
            image_map[frame] = file

    return image_map


# ============================================================
# FIXTURE TEST
# ============================================================

def points_inside_fixture(x, y):

    points = np.column_stack([
        x,
        y
    ]).astype(np.float32)

    result = np.zeros(
        len(points),
        dtype=bool
    )

    for i, point in enumerate(points):

        result[i] = (
            cv2.pointPolygonTest(
                FIXTURE_POLYGON.astype(
                    np.float32
                ),
                (
                    float(point[0]),
                    float(point[1])
                ),
                False
            ) >= 0
        )

    return result


# ============================================================
# LOADING ZONE TEST
# ============================================================

def points_inside_loading_zone(x, y):

    return (
        (x >= LOADING_X_MIN) &
        (x <= LOADING_X_MAX) &
        (y >= LOADING_Y_MIN) &
        (y <= LOADING_Y_MAX)
    )


# ============================================================
# REMOVE RIGID TRANSLATION + ROTATION
#
# Rigid motion:
#
# u = Tx - theta*y
# v = Ty + theta*x
#
# We estimate these terms using DIC points away from
# the loading region.
# ============================================================

def remove_rigid_motion(df):

    x = df["X"].values.astype(float)

    y = df["Y"].values.astype(float)

    u = df["dx"].values.astype(float)

    v = df["dy"].values.astype(float)

    loading = points_inside_loading_zone(
        x,
        y
    )

    fixture = points_inside_fixture(
        x,
        y
    )

    valid = (
        ~loading &
        ~fixture &
        np.isfinite(u) &
        np.isfinite(v)
    )

    if np.sum(valid) < 20:

        print(
            "Not enough points for rigid-motion correction."
        )

        return df.copy()

    # --------------------------------------------------------
    # Center coordinates
    # --------------------------------------------------------

    x0 = np.mean(
        x[valid]
    )

    y0 = np.mean(
        y[valid]
    )

    xc = x - x0
    yc = y - y0

    # --------------------------------------------------------
    # Rigid model:
    #
    # u = Tx - theta*y
    # v = Ty + theta*x
    #
    # Combine both equations.
    # --------------------------------------------------------

    A = np.zeros(
        (2 * np.sum(valid), 3)
    )

    b = np.zeros(
        2 * np.sum(valid)
    )

    xv = xc[valid]
    yv = yc[valid]

    uv = u[valid]
    vv = v[valid]

    n = len(xv)

    A[:n, 0] = 1
    A[:n, 2] = -yv

    A[n:, 1] = 1
    A[n:, 2] = xv

    b[:n] = uv
    b[n:] = vv

    params, _, _, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None
    )

    tx = params[0]
    ty = params[1]
    theta = params[2]

    # --------------------------------------------------------
    # Rigid displacement field
    # --------------------------------------------------------

    rigid_u = (
        tx -
        theta * yc
    )

    rigid_v = (
        ty +
        theta * xc
    )

    corrected = df.copy()

    corrected["dx_corrected"] = (
        u - rigid_u
    )

    corrected["dy_corrected"] = (
        v - rigid_v
    )

    print(
        f"Rigid translation: Tx={tx:.5f}, Ty={ty:.5f}"
    )

    print(
        f"Rigid rotation: theta={theta:.8f}"
    )

    return corrected


# ============================================================
# LOCAL AFFINE STRAIN
#
# Around each DIC point we fit:
#
# u = a0 + a1*dx + a2*dy
# v = b0 + b1*dx + b2*dy
#
# The displacement gradients are:
#
# du/dx = a1
# du/dy = a2
# dv/dx = b1
# dv/dy = b2
#
# Then principal strain is calculated.
# ============================================================

def calculate_local_strain(df):

    x = df["X"].values.astype(float)
    y = df["Y"].values.astype(float)

    u = df["dx_corrected"].values.astype(float)
    v = df["dy_corrected"].values.astype(float)

    valid = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(u) &
        np.isfinite(v)
    )

    x = x[valid]
    y = y[valid]
    u = u[valid]
    v = v[valid]

    points = np.column_stack([
        x,
        y
    ])

    tree = cKDTree(
        points
    )

    strain_values = []

    valid_x = []
    valid_y = []

    for i in range(
        len(points)
    ):

        neighbours = tree.query_ball_point(
            points[i],
            LOCAL_RADIUS
        )

        if len(neighbours) < MIN_NEIGHBORS:
            continue

        neighbours = np.asarray(
            neighbours,
            dtype=int
        )

        # Local coordinates
        dx = (
            x[neighbours] -
            x[i]
        )

        dy = (
            y[neighbours] -
            y[i]
        )

        du = (
            u[neighbours] -
            u[i]
        )

        dv = (
            v[neighbours] -
            v[i]
        )

        # ----------------------------------------------------
        # Local affine fit
        # ----------------------------------------------------

        A = np.column_stack([
            np.ones(len(neighbours)),
            dx,
            dy
        ])

        try:

            coeff_u, _, _, _ = np.linalg.lstsq(
                A,
                du,
                rcond=None
            )

            coeff_v, _, _, _ = np.linalg.lstsq(
                A,
                dv,
                rcond=None
            )

        except np.linalg.LinAlgError:

            continue

        du_dx = coeff_u[1]
        du_dy = coeff_u[2]

        dv_dx = coeff_v[1]
        dv_dy = coeff_v[2]

        # ----------------------------------------------------
        # Small-strain tensor
        # ----------------------------------------------------

        exx = du_dx

        eyy = dv_dy

        exy = 0.5 * (
            du_dy +
            dv_dx
        )

        # ----------------------------------------------------
        # Principal strains
        # ----------------------------------------------------

        mean = (
            exx +
            eyy
        ) / 2

        radius = np.sqrt(
            (
                (exx - eyy) / 2
            ) ** 2
            +
            exy ** 2
        )

        p1 = mean + radius
        p2 = mean - radius

        principal = max(
            abs(p1),
            abs(p2)
        )

        valid_x.append(
            x[i]
        )

        valid_y.append(
            y[i]
        )

        strain_values.append(
            principal
        )

    result = pd.DataFrame({
        "X": valid_x,
        "Y": valid_y,
        "Local_Principal_Strain":
            strain_values
    })

    return result


# ============================================================
# CREATE GRID FOR DISPLAY
# ============================================================

def create_display_grid(
    strain_df,
    width,
    height
):

    x = strain_df["X"].values
    y = strain_df["Y"].values
    s = strain_df[
        "Local_Principal_Strain"
    ].values

    grid_x_values = np.arange(
        0,
        width,
        GRID_STEP
    )

    grid_y_values = np.arange(
        0,
        height,
        GRID_STEP
    )

    grid_x, grid_y = np.meshgrid(
        grid_x_values,
        grid_y_values
    )

    grid_s = griddata(
        (
            x,
            y
        ),
        s,
        (
            grid_x,
            grid_y
        ),
        method="linear"
    )

    # Fill only interpolation holes
    # using nearest values.
    missing = np.isnan(
        grid_s
    )

    if np.any(missing):

        grid_s[missing] = griddata(
            (
                x,
                y
            ),
            s,
            (
                grid_x[missing],
                grid_y[missing]
            ),
            method="nearest"
        )

    # Smooth final strain field
    grid_s = gaussian_filter(
        grid_s,
        sigma=STRAIN_SMOOTH_SIGMA
    )

    return (
        grid_x,
        grid_y,
        grid_s
    )


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    image,
    grid_x,
    grid_y,
    grid_s,
    interval_name,
    height,
    width
):

    # --------------------------------------------------------
    # Actual fixture mask
    # --------------------------------------------------------

    fixture_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        fixture_mask,
        [FIXTURE_POLYGON],
        1
    )

    gx = np.clip(
        grid_x.astype(int),
        0,
        width - 1
    )

    gy = np.clip(
        grid_y.astype(int),
        0,
        height - 1
    )

    grid_fixture = (
        fixture_mask[gy, gx] == 1
    )

    display = grid_s.copy()

    display[
        grid_fixture
    ] = np.nan

    # --------------------------------------------------------
    # Remove extreme outliers from colour scale
    # --------------------------------------------------------

    valid = display[
        np.isfinite(display)
    ]

    if len(valid) == 0:
        return

    vmax = np.percentile(
        valid,
        99
    )

    if vmax <= 0:
        vmax = np.max(
            valid
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(17, 7)
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    heat = ax.pcolormesh(
        grid_x,
        grid_y,
        display,
        cmap=DIC_CMAP,
        shading="auto",
        alpha=0.68,
        vmin=0,
        vmax=vmax
    )

    # Fixture outline
    polygon = np.vstack([
        FIXTURE_POLYGON,
        FIXTURE_POLYGON[0]
    ])

    ax.plot(
        polygon[:, 0],
        polygon[:, 1],
        "k-",
        linewidth=2
    )

    cbar = fig.colorbar(
        heat,
        ax=ax
    )

    cbar.set_label(
        "Maximum principal strain"
    )

    ax.set_title(
        f"Local DIC strain heatmap — {interval_name}"
    )

    ax.set_xlabel(
        "X (pixels)"
    )

    ax.set_ylabel(
        "Y (pixels)"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"heatmap_{interval_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


# ============================================================
# PROCESS ONE INTERVAL
# ============================================================

def process_interval(
    start_frame,
    end_frame,
    image_map
):

    print("\n========================================")
    print(
        f"Processing {start_frame} -> {end_frame}"
    )
    print("========================================")

    dic_file = os.path.join(
        DIC_DIR,
        f"dic_{start_frame}_{end_frame}.csv"
    )

    if not os.path.exists(dic_file):

        print(
            "DIC file missing:",
            dic_file
        )

        return None

    if end_frame not in image_map:

        print(
            "Image missing:",
            end_frame
        )

        return None

    # --------------------------------------------------------
    # Read DIC
    # --------------------------------------------------------

    df = pd.read_csv(
        dic_file
    )

    print(
        "Original DIC points:",
        len(df)
    )

    # --------------------------------------------------------
    # ZNCC
    # --------------------------------------------------------

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ].copy()

    print(
        "After ZNCC filtering:",
        len(df)
    )

    # --------------------------------------------------------
    # Remove rigid motion
    # --------------------------------------------------------

    df = remove_rigid_motion(
        df
    )

    # --------------------------------------------------------
    # Local strain
    # --------------------------------------------------------

    strain_df = calculate_local_strain(
        df
    )

    print(
        "Local strain points:",
        len(strain_df)
    )

    if len(strain_df) < 100:

        print(
            "Too few strain points."
        )

        return None

    # --------------------------------------------------------
    # Remove fixture points from strain results
    # --------------------------------------------------------

    fixture = points_inside_fixture(
        strain_df["X"].values,
        strain_df["Y"].values
    )

    strain_df = strain_df[
        ~fixture
    ].copy()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    values = strain_df[
        "Local_Principal_Strain"
    ].values

    values = values[
        np.isfinite(values)
    ]

    mean_value = np.mean(
        values
    )

    p95 = np.percentile(
        values,
        95
    )

    p99 = np.percentile(
        values,
        99
    )

    maximum = np.max(
        values
    )

    # --------------------------------------------------------
    # Find maximum outside broad loading region
    # --------------------------------------------------------

    loading = points_inside_loading_zone(
        strain_df["X"].values,
        strain_df["Y"].values
    )

    outside = strain_df[
        ~loading
    ].copy()

    outside_values = outside[
        "Local_Principal_Strain"
    ].values

    outside_values = outside_values[
        np.isfinite(outside_values)
    ]

    outside_max = np.max(
        outside_values
    )

    max_index = np.argmax(
        outside[
            "Local_Principal_Strain"
        ].values
    )

    max_x = outside.iloc[
        max_index
    ]["X"]

    max_y = outside.iloc[
        max_index
    ]["Y"]

    # --------------------------------------------------------
    # Create grid
    # --------------------------------------------------------

    image = cv2.imread(
        image_map[end_frame],
        cv2.IMREAD_GRAYSCALE
    )

    height, width = image.shape

    (
        grid_x,
        grid_y,
        grid_s
    ) = create_display_grid(
        strain_df,
        width,
        height
    )

    interval_name = (
        f"{start_frame}_{end_frame}"
    )

    create_heatmap(
        image,
        grid_x,
        grid_y,
        grid_s,
        interval_name,
        height,
        width
    )

    # --------------------------------------------------------
    # Save local strain
    # --------------------------------------------------------

    strain_file = os.path.join(
        OUTPUT_DIR,
        f"local_strain_{interval_name}.csv"
    )

    strain_df.to_csv(
        strain_file,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nLOCAL STRAIN STATISTICS")

    print(
        f"Mean strain       : {mean_value:.8f}"
    )

    print(
        f"95th percentile   : {p95:.8f}"
    )

    print(
        f"99th percentile   : {p99:.8f}"
    )

    print(
        f"Maximum overall   : {maximum:.8f}"
    )

    print(
        f"Maximum outside loading zone: "
        f"{outside_max:.8f}"
    )

    print(
        f"Maximum location outside zone: "
        f"X={max_x:.1f}, Y={max_y:.1f}"
    )

    return {
        "Frame_Start": start_frame,
        "Frame_End": end_frame,
        "DIC_Points": len(df),
        "Strain_Points": len(strain_df),
        "Mean_Strain": mean_value,
        "P95_Strain": p95,
        "P99_Strain": p99,
        "Maximum_Strain": maximum,
        "Outside_Loading_Max": outside_max,
        "Maximum_X": max_x,
        "Maximum_Y": max_y
    }


# ============================================================
# MAIN
# ============================================================

print("==============================================")
print(" STAGE 7 - LOCAL AFFINE DIC STRAIN")
print("==============================================")

image_map = load_image_map()

print(
    "\nAvailable ROI frames:",
    len(image_map)
)

print(
    "First frame:",
    min(image_map.keys())
)

print(
    "Last frame:",
    max(image_map.keys())
)

summary = []

for start_frame, end_frame in INTERVALS:

    result = process_interval(
        start_frame,
        end_frame,
        image_map
    )

    if result is not None:
        summary.append(result)


# ============================================================
# SAVE SUMMARY
# ============================================================

if summary:

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "stage7_local_strain_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n==============================================")
    print("STAGE 7 SUMMARY")
    print("==============================================")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        summary_file
    )


print("\n==============================================")
print("STAGE 7 COMPLETE")
print("==============================================")