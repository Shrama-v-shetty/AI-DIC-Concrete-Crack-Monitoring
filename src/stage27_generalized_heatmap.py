import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "BGSL_ROI"
)

STAGE9_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage9_Full_Sequential_DIC"
)

STAGE24_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage24_Rolling_Reference_Heatmap"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "Stage27_Generalized_Heatmap"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

ZNCC_THRESHOLD = 0.95

NEIGHBORS = 25

MAX_RADIUS = 100

GRID_STEP = 5

VISUAL_SMOOTHING = 1.2


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generalized DIC spatial strain heatmap"
)

parser.add_argument(
    "--reference",
    type=int,
    default=1274,
    help="Reference frame number"
)

parser.add_argument(
    "--target",
    type=int,
    default=1374,
    help="Target frame number"
)

args = parser.parse_args()

reference_frame = args.reference
target_frame = args.target


# ============================================================
# HELPER: IMAGE FRAME NUMBER
# ============================================================

def image_path(frame):

    filename = (
        f"BGSL-2-{frame:04d}_0.tif"
    )

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    # Fallback search in case filename formatting changes
    for name in os.listdir(IMAGE_DIR):

        if not name.lower().endswith(".tif"):
            continue

        numbers = re.findall(
            r"\d+",
            name
        )

        if len(numbers) >= 2:

            try:
                detected = int(
                    numbers[-2]
                )

                if detected == frame:
                    return os.path.join(
                        IMAGE_DIR,
                        name
                    )

            except ValueError:
                pass

    return None


# ============================================================
# HELPER: DISCOVER STAGE9 PAIRS
# ============================================================

def discover_stage9_pairs():

    pairs = {}

    pattern = re.compile(
        r"dic_(\d+)_(\d+)\.csv$",
        re.IGNORECASE
    )

    if not os.path.isdir(STAGE9_DIR):
        return pairs

    for filename in os.listdir(STAGE9_DIR):

        match = pattern.match(filename)

        if match:

            f1 = int(match.group(1))
            f2 = int(match.group(2))

            pairs[(f1, f2)] = os.path.join(
                STAGE9_DIR,
                filename
            )

    return pairs


# ============================================================
# HELPER: LOAD A DIC FILE
# ============================================================

def load_dic(path):

    df = pd.read_csv(path)

    required = [
        "X",
        "Y",
        "dx",
        "dy",
        "ZNCC"
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"DIC file does not contain "
                f"required column: {col}\n"
                f"{path}"
            )

    df = df[
        required
    ].copy()

    df = df[
        np.isfinite(df["X"]) &
        np.isfinite(df["Y"]) &
        np.isfinite(df["dx"]) &
        np.isfinite(df["dy"]) &
        np.isfinite(df["ZNCC"])
    ]

    df = df[
        df["ZNCC"] >= ZNCC_THRESHOLD
    ]

    # Remove duplicate spatial coordinates
    df = (
        df.groupby(
            ["X", "Y"],
            as_index=False
        )
        .agg({
            "dx": "median",
            "dy": "median",
            "ZNCC": "median"
        })
    )

    return df


# ============================================================
# LOAD EXISTING ROLLING DIC
# ============================================================

def find_existing_rolling_dic(
    reference,
    target
):

    filename = (
        f"dic_{reference}_{target}.csv"
    )

    path = os.path.join(
        STAGE24_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# BUILD CUMULATIVE DIC FROM STAGE9
# ============================================================

def build_cumulative_dic(
    reference,
    target,
    stage9_pairs
):

    if target <= reference:

        raise ValueError(
            "Target frame must be greater "
            "than reference frame."
        )

    # --------------------------------------------------------
    # Find all required sequential pairs.
    #
    # We use only pairs that actually exist.
    # This prevents us from assuming that missing
    # image numbers are valid consecutive frames.
    # --------------------------------------------------------

    required_pairs = []

    current = reference

    while current < target:

        candidates = [
            pair
            for pair in stage9_pairs
            if pair[0] == current
            and pair[1] <= target
        ]

        if not candidates:

            raise RuntimeError(
                "\nCannot build cumulative DIC.\n"
                f"No Stage9 pair beginning at frame "
                f"{current} was found.\n\n"
                "This usually means the requested "
                "reference → target range crosses "
                "a missing image section."
            )

        # Normally there should be one consecutive pair.
        next_pair = min(
            candidates,
            key=lambda p: p[1]
        )

        if next_pair[1] != current + 1:

            raise RuntimeError(
                "\nThere is a gap in the DIC sequence.\n"
                f"Expected pair starting at {current}, "
                f"but found {next_pair}."
            )

        required_pairs.append(
            next_pair
        )

        current = next_pair[1]

    print(
        f"\nSequential DIC pairs required: "
        f"{len(required_pairs)}"
    )

    # --------------------------------------------------------
    # First pair
    # --------------------------------------------------------

    first_pair = required_pairs[0]

    first_df = load_dic(
        stage9_pairs[first_pair]
    )

    coordinates = first_df[
        ["X", "Y"]
    ].to_numpy(
        dtype=float
    )

    cumulative_dx = first_df[
        "dx"
    ].to_numpy(
        dtype=float
    )

    cumulative_dy = first_df[
        "dy"
    ].to_numpy(
        dtype=float
    )

    cumulative_zncc = first_df[
        "ZNCC"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Add subsequent sequential displacements
    #
    # This is appropriate for the small incremental
    # displacement fields generated by the sequential DIC.
    # --------------------------------------------------------

    for pair_number, pair in enumerate(
        required_pairs[1:],
        start=2
    ):

        path = stage9_pairs[pair]

        pair_df = load_dic(path)

        pair_df = pair_df.set_index(
            ["X", "Y"]
        )

        new_dx = []
        new_dy = []
        new_zncc = []

        for x, y in coordinates:

            key = (
                x,
                y
            )

            if key in pair_df.index:

                row = pair_df.loc[key]

                new_dx.append(
                    float(row["dx"])
                )

                new_dy.append(
                    float(row["dy"])
                )

                new_zncc.append(
                    float(row["ZNCC"])
                )

            else:

                new_dx.append(
                    np.nan
                )

                new_dy.append(
                    np.nan
                )

                new_zncc.append(
                    np.nan
                )

        new_dx = np.asarray(
            new_dx,
            dtype=float
        )

        new_dy = np.asarray(
            new_dy,
            dtype=float
        )

        new_zncc = np.asarray(
            new_zncc,
            dtype=float
        )

        valid = (
            np.isfinite(new_dx)
            &
            np.isfinite(new_dy)
        )

        cumulative_dx[valid] += (
            new_dx[valid]
        )

        cumulative_dy[valid] += (
            new_dy[valid]
        )

        cumulative_zncc[valid] = np.minimum(
            cumulative_zncc[valid],
            new_zncc[valid]
        )

        print(
            f"Accumulated pair "
            f"{pair_number}/{len(required_pairs)}: "
            f"{pair[0]} → {pair[1]}"
        )

    result = pd.DataFrame({
        "X": coordinates[:, 0],
        "Y": coordinates[:, 1],
        "dx": cumulative_dx,
        "dy": cumulative_dy,
        "ZNCC": cumulative_zncc
    })

    result = result[
        np.isfinite(result["dx"])
        &
        np.isfinite(result["dy"])
        &
        np.isfinite(result["ZNCC"])
    ]

    result = result[
        result["ZNCC"] >= ZNCC_THRESHOLD
    ]

    return result


# ============================================================
# LOCAL AFFINE STRAIN
# ============================================================

def calculate_local_strain(df):

    coordinates = df[
        ["X", "Y"]
    ].to_numpy(
        dtype=float
    )

    u = df[
        "dx"
    ].to_numpy(
        dtype=float
    )

    v = df[
        "dy"
    ].to_numpy(
        dtype=float
    )

    zncc = df[
        "ZNCC"
    ].to_numpy(
        dtype=float
    )

    tree = cKDTree(
        coordinates
    )

    results = []

    total = len(coordinates)

    for i, point in enumerate(
        coordinates
    ):

        distances, indices = tree.query(
            point,
            k=min(
                NEIGHBORS,
                total
            )
        )

        distances = np.atleast_1d(
            distances
        )

        indices = np.atleast_1d(
            indices
        )

        valid = (
            distances <= MAX_RADIUS
        )

        distances = distances[valid]
        indices = indices[valid]

        if len(indices) < 8:
            continue

        local_xy = coordinates[
            indices
        ]

        x_local = (
            local_xy[:, 0]
            - point[0]
        )

        y_local = (
            local_xy[:, 1]
            - point[1]
        )

        local_u = u[
            indices
        ]

        local_v = v[
            indices
        ]

        A = np.column_stack([
            np.ones(len(indices)),
            x_local,
            y_local
        ])

        # Closer DIC points receive more weight.
        weights = 1.0 / (
            distances + 5.0
        )

        W = np.sqrt(
            weights
        )

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

        exx = du_dx

        eyy = dv_dy

        exy = 0.5 * (
            du_dy + dv_dx
        )

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
            "ZNCC": zncc[i],
            "Neighbors_Used": len(indices),
            "Neighborhood_Radius": np.max(
                distances
            )
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

print("\n==============================================")
print("STAGE 27 - GENERALIZED DIC HEATMAP")
print("==============================================")

print(
    f"\nReference frame : {reference_frame}"
)

print(
    f"Target frame    : {target_frame}"
)


# ============================================================
# CHECK IMAGES
# ============================================================

reference_image = image_path(
    reference_frame
)

target_image = image_path(
    target_frame
)

if reference_image is None:

    raise FileNotFoundError(
        f"\nReference image not found: "
        f"{reference_frame}"
    )

if target_image is None:

    raise FileNotFoundError(
        f"\nTarget image not found: "
        f"{target_frame}"
    )

print(
    "\nReference image:",
    reference_image
)

print(
    "Target image:",
    target_image
)


# ============================================================
# FIND DIC
# ============================================================

print("\nSearching for DIC data...")

existing = find_existing_rolling_dic(
    reference_frame,
    target_frame
)

if existing is not None:

    print(
        "\nExisting rolling-reference DIC found:"
    )

    print(existing)

    dic_df = load_dic(
        existing
    )

else:

    print(
        "\nNo direct rolling-reference file found."
    )

    print(
        "Searching Stage9 sequential DIC..."
    )

    stage9_pairs = (
        discover_stage9_pairs()
    )

    print(
        f"Stage9 DIC pairs found: "
        f"{len(stage9_pairs)}"
    )

    dic_df = build_cumulative_dic(
        reference_frame,
        target_frame,
        stage9_pairs
    )


print(
    "\nFinal DIC points:",
    len(dic_df)
)


# ============================================================
# CALCULATE LOCAL STRAIN
# ============================================================

print(
    "\nCalculating local affine strain..."
)

strain_df = calculate_local_strain(
    dic_df
)

if strain_df.empty:

    raise RuntimeError(
        "No valid local strain points."
    )

print(
    "Local strain points:",
    len(strain_df)
)


# ============================================================
# SAVE POINT-LEVEL STRAIN
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    f"local_strain_{reference_frame}_{target_frame}.csv"
)

strain_df.to_csv(
    output_csv,
    index=False
)

print(
    "\nSaved:",
    output_csv
)


# ============================================================
# LOAD TARGET IMAGE
# ============================================================

image = cv2.imread(
    target_image,
    cv2.IMREAD_GRAYSCALE
)

if image is None:

    raise RuntimeError(
        "Target image could not be loaded."
    )

height, width = image.shape


# ============================================================
# VISUALIZATION GRID
# ============================================================

x_min = int(
    strain_df["X"].min()
)

x_max = min(
    width - 1,
    int(strain_df["X"].max())
)

y_min = int(
    strain_df["Y"].min()
)

y_max = min(
    height - 1,
    int(strain_df["Y"].max())
)

x_values = np.arange(
    x_min,
    x_max + 1,
    GRID_STEP
)

y_values = np.arange(
    y_min,
    y_max + 1,
    GRID_STEP
)

grid_x, grid_y = np.meshgrid(
    x_values,
    y_values
)


# ============================================================
# INTERPOLATE LOCAL STRAIN
# ============================================================

print(
    "\nInterpolating local strain..."
)

points = strain_df[
    ["X", "Y"]
].to_numpy()

values = strain_df[
    "Principal_Strain"
].to_numpy()

strain_grid = griddata(
    points,
    values,
    (grid_x, grid_y),
    method="linear"
)


# ============================================================
# VISUAL SMOOTHING
# ============================================================

valid_mask = np.isfinite(
    strain_grid
)

if not np.any(valid_mask):

    raise RuntimeError(
        "Interpolation produced no valid heatmap field."
    )

filled = strain_grid.copy()

median_value = np.nanmedian(
    filled
)

filled[
    ~valid_mask
] = median_value

smooth_grid = gaussian_filter(
    filled,
    sigma=VISUAL_SMOOTHING
)

smooth_grid[
    ~valid_mask
] = np.nan


# ============================================================
# COLOR RANGE
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
# STATISTICS
# ============================================================

print("\n==============================================")
print("STRAIN STATISTICS")
print("==============================================")

print(
    f"Minimum : {np.min(valid):.8f}"
)

print(
    f"Maximum : {np.max(valid):.8f}"
)

print(
    f"Mean    : {np.mean(valid):.8f}"
)

print(
    f"P95     : {np.percentile(valid, 95):.8f}"
)

print(
    f"P99     : {np.percentile(valid, 99):.8f}"
)


# ============================================================
# HEATMAP
# ============================================================

print(
    "\nGenerating final heatmap..."
)

output_png = os.path.join(
    OUTPUT_DIR,
    f"dic_heatmap_{reference_frame}_{target_frame}.png"
)

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
    f"Local Affine DIC Spatial Strain Heatmap\n"
    f"Frame {reference_frame} → {target_frame}"
)

plt.xlabel(
    "X (pixels)"
)

plt.ylabel(
    "Y (pixels)"
)

plt.tight_layout()

plt.savefig(
    output_png,
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print("STAGE 27 COMPLETE")
print("==============================================")

print(
    "\nHeatmap:"
)

print(
    output_png
)

print(
    "\nStrain CSV:"
)

print(
    output_csv
)