import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# STAGE 15 - SPATIAL EVENT VALIDATION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "BGSL_ROI"
)

DIC_DIR = os.path.join(
    BASE_DIR,
    "Stage9_Full_Sequential_DIC"
)

EVENT_CSV = os.path.join(
    BASE_DIR,
    "Stage14_Strong_Event_Verification",
    "verified_strong_events.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage15_Spatial_Event_Validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

GRID_STEP = 10
GAUSSIAN_SIGMA = 2

FIXTURE_X1 = 930
FIXTURE_X2 = 1080
FIXTURE_Y1 = 450
FIXTURE_Y2 = 638


# ============================================================
# FRAME NUMBER EXTRACTION
# ============================================================

def extract_frame_number(filename):

    numbers = re.findall(
        r"\d+",
        os.path.basename(filename)
    )

    if len(numbers) >= 2:
        return int(numbers[-2])

    return None


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(frame):

    for name in os.listdir(IMAGE_DIR):

        if not name.lower().endswith(
            (
                ".tif",
                ".tiff",
                ".png",
                ".jpg",
                ".jpeg"
            )
        ):
            continue

        if extract_frame_number(name) == frame:
            return os.path.join(
                IMAGE_DIR,
                name
            )

    return None


# ============================================================
# FIND DIC CSV
# ============================================================

def find_dic_csv(
    frame1,
    frame2
):

    filename = (
        f"dic_{frame1}_{frame2}.csv"
    )

    path = os.path.join(
        DIC_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# DIC -> PRINCIPAL STRAIN
# ============================================================

def calculate_strain(df):

    rename = {}

    for column in df.columns:

        name = column.strip().lower()

        if name == "x":
            rename[column] = "X"

        elif name == "y":
            rename[column] = "Y"

        elif name == "dx":
            rename[column] = "dx"

        elif name == "dy":
            rename[column] = "dy"

    df = df.rename(
        columns=rename
    )

    required = [
        "X",
        "Y",
        "dx",
        "dy"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    x = pd.to_numeric(
        df["X"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        df["Y"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    dx = pd.to_numeric(
        df["dx"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    dy = pd.to_numeric(
        df["dy"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(dx)
        & np.isfinite(dy)
    )

    x = x[valid]
    y = y[valid]
    dx = dx[valid]
    dy = dy[valid]

    if len(x) < 50:

        raise RuntimeError(
            f"Too few valid DIC points: {len(x)}"
        )

    x_values = np.arange(
        np.floor(x.min()),
        np.ceil(x.max())
        + GRID_STEP,
        GRID_STEP
    )

    y_values = np.arange(
        np.floor(y.min()),
        np.ceil(y.max())
        + GRID_STEP,
        GRID_STEP
    )

    gx, gy = np.meshgrid(
        x_values,
        y_values
    )

    gdx = griddata(
        (x, y),
        dx,
        (gx, gy),
        method="linear"
    )

    gdy = griddata(
        (x, y),
        dy,
        (gx, gy),
        method="linear"
    )

    # Fill interpolation gaps
    for field, values in [
        (gdx, dx),
        (gdy, dy)
    ]:

        bad = ~np.isfinite(field)

        if np.any(bad):

            field[bad] = griddata(
                (x, y),
                values,
                (gx[bad], gy[bad]),
                method="nearest"
            )

    gdx = gaussian_filter(
        gdx,
        sigma=GAUSSIAN_SIGMA
    )

    gdy = gaussian_filter(
        gdy,
        sigma=GAUSSIAN_SIGMA
    )

    du_dy, du_dx = np.gradient(
        gdx,
        GRID_STEP,
        GRID_STEP
    )

    dv_dy, dv_dx = np.gradient(
        gdy,
        GRID_STEP,
        GRID_STEP
    )

    exx = du_dx

    eyy = dv_dy

    exy = 0.5 * (
        du_dy + dv_dx
    )

    mean_strain = 0.5 * (
        exx + eyy
    )

    radius = np.sqrt(
        ((exx - eyy) * 0.5) ** 2
        + exy ** 2
    )

    principal_strain = (
        mean_strain
        + radius
    )

    return (
        gx,
        gy,
        principal_strain
    )


# ============================================================
# HEATMAP COLOR MAP
# ============================================================

def create_colormap():

    return LinearSegmentedColormap.from_list(
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
# LOAD EVENTS
# ============================================================

print("=" * 75)

print(
    "STAGE 15 - SPATIAL EVENT VALIDATION"
)

print("=" * 75)

if not os.path.exists(EVENT_CSV):

    raise FileNotFoundError(
        f"Event file not found:\n{EVENT_CSV}"
    )

events = pd.read_csv(
    EVENT_CSV
)

print()
print(
    f"Events loaded: {len(events)}"
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH EVENT
# ============================================================

for index, event in events.iterrows():

    frame1 = int(
        event["Frame_1"]
    )

    frame2 = int(
        event["Frame_2"]
    )

    print()
    print(
        "=" * 65
    )

    print(
        f"EVENT {index + 1}: "
        f"{frame1} -> {frame2}"
    )

    image_path = find_image(
        frame2
    )

    dic_path = find_dic_csv(
        frame1,
        frame2
    )

    print(
        "End image:",
        image_path
    )

    print(
        "DIC CSV:",
        dic_path
    )

    if (
        image_path is None
        or dic_path is None
    ):

        print(
            "Skipping event: "
            "required file not found."
        )

        continue


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "Skipping event: image could not be read."
        )

        continue


    # --------------------------------------------------------
    # READ DIC
    # --------------------------------------------------------

    dic = pd.read_csv(
        dic_path
    )

    gx, gy, strain = calculate_strain(
        dic
    )


    # --------------------------------------------------------
    # REMOVE FIXTURE FOR STATISTICS
    # --------------------------------------------------------

    statistic_mask = ~(
        (gx >= FIXTURE_X1)
        & (gx <= FIXTURE_X2)
        & (gy >= FIXTURE_Y1)
        & (gy <= FIXTURE_Y2)
    )

    valid = (
        np.isfinite(strain)
        & statistic_mask
    )

    values = strain[
        valid
    ]

    if values.size == 0:

        print(
            "Skipping event: "
            "no valid strain field."
        )

        continue


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    mean_strain = float(
        np.mean(values)
    )

    p95 = float(
        np.percentile(
            values,
            95
        )
    )

    p99 = float(
        np.percentile(
            values,
            99
        )
    )

    max_strain = float(
        np.max(values)
    )


    # --------------------------------------------------------
    # HIGH-STRAIN REGION
    # --------------------------------------------------------

    high_mask = (
        valid
        & (strain >= p99)
    )

    high_y, high_x = np.where(
        high_mask
    )

    if len(high_x) > 0:

        centroid_x = float(
            np.mean(
                gx[
                    high_y,
                    high_x
                ]
            )
        )

        centroid_y = float(
            np.mean(
                gy[
                    high_y,
                    high_x
                ]
            )
        )

        x_min = float(
            np.min(
                gx[
                    high_y,
                    high_x
                ]
            )
        )

        x_max = float(
            np.max(
                gx[
                    high_y,
                    high_x
                ]
            )
        )

        y_min = float(
            np.min(
                gy[
                    high_y,
                    high_x
                ]
            )
        )

        y_max = float(
            np.max(
                gy[
                    high_y,
                    high_x
                ]
            )
        )

    else:

        centroid_x = np.nan
        centroid_y = np.nan

        x_min = x_max = np.nan
        y_min = y_max = np.nan


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        "STRAIN"
    )

    print(
        f"Mean : {mean_strain:.6f}"
    )

    print(
        f"P95  : {p95:.6f}"
    )

    print(
        f"P99  : {p99:.6f}"
    )

    print(
        f"Max  : {max_strain:.6f}"
    )

    print()

    print(
        "HIGH-STRAIN REGION"
    )

    print(
        f"Cells    : {len(high_x)}"
    )

    print(
        f"Centroid : "
        f"({centroid_x:.1f}, "
        f"{centroid_y:.1f})"
    )

    print(
        f"X range  : "
        f"{x_min:.1f} -> "
        f"{x_max:.1f}"
    )

    print(
        f"Y range  : "
        f"{y_min:.1f} -> "
        f"{y_max:.1f}"
    )


    # --------------------------------------------------------
    # SAVE VALIDATION FIGURE
    # --------------------------------------------------------

    cmap = create_colormap()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6)
    )


    # Original
    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        f"Original frame {frame2}"
    )

    axes[0].set_xlabel(
        "X"
    )

    axes[0].set_ylabel(
        "Y"
    )


    # Full heatmap
    axes[1].imshow(
        image,
        cmap="gray"
    )

    axes[1].imshow(
        strain,
        extent=[
            gx.min(),
            gx.max(),
            gy.max(),
            gy.min()
        ],
        cmap=cmap,
        alpha=0.60,
        vmin=0,
        vmax=p99
    )

    axes[1].add_patch(
        plt.Rectangle(
            (
                FIXTURE_X1,
                FIXTURE_Y1
            ),
            FIXTURE_X2 - FIXTURE_X1,
            FIXTURE_Y2 - FIXTURE_Y1,
            fill=False,
            linestyle="--",
            linewidth=2
        )
    )

    if np.isfinite(
        centroid_x
    ):

        axes[1].scatter(
            [centroid_x],
            [centroid_y],
            marker="x",
            s=120,
            linewidths=2
        )

    axes[1].set_title(
        "DIC principal strain"
    )

    axes[1].set_xlabel(
        "X"
    )

    axes[1].set_ylabel(
        "Y"
    )


    # Central region
    axes[2].imshow(
        image,
        cmap="gray"
    )

    axes[2].imshow(
        strain,
        extent=[
            gx.min(),
            gx.max(),
            gy.max(),
            gy.min()
        ],
        cmap=cmap,
        alpha=0.65,
        vmin=0,
        vmax=p99
    )

    axes[2].set_xlim(
        max(
            0,
            FIXTURE_X1 - 350
        ),
        min(
            image.shape[1],
            FIXTURE_X2 + 350
        )
    )

    axes[2].set_ylim(
        min(
            image.shape[0],
            FIXTURE_Y2 + 100
        ),
        max(
            0,
            FIXTURE_Y1 - 400
        )
    )

    axes[2].set_title(
        "Central deformation region"
    )

    axes[2].set_xlabel(
        "X"
    )

    axes[2].set_ylabel(
        "Y"
    )


    # Shared colorbar
    scalar = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=0,
            vmax=p99
        )
    )

    scalar.set_array([])

    fig.colorbar(
        scalar,
        ax=axes,
        shrink=0.80,
        label="Principal strain"
    )

    fig.suptitle(
        f"Spatial Event Validation: "
        f"{frame1} -> {frame2}"
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.92,
        bottom=0.10,
        top=0.88,
        wspace=0.20
    )


    output_image = os.path.join(
        OUTPUT_DIR,
        f"event_{frame1}_{frame2}_validation.png"
    )

    fig.savefig(
        output_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "Event_Number":
                index + 1,

            "Frame_1":
                frame1,

            "Frame_2":
                frame2,

            "Time_1":
                event["Time_1"],

            "Time_2":
                event["Time_2"],

            "Load_Delta_kN":
                event["Load_Delta_kN"],

            "Ch7_Displacement_Delta_um":
                event[
                    "Primary_Displacement_Delta_um"
                ],

            "DIC_Average_Magnitude_px":
                event[
                    "DIC_Average_Magnitude"
                ],

            "DIC_Maximum_Magnitude_px":
                event[
                    "DIC_Maximum_Magnitude"
                ],

            "DIC_Average_ZNCC":
                event[
                    "DIC_Average_ZNCC"
                ],

            "Principal_Strain_Mean":
                mean_strain,

            "Principal_Strain_P95":
                p95,

            "Principal_Strain_P99":
                p99,

            "Principal_Strain_Max":
                max_strain,

            "High_Strain_Cells":
                len(high_x),

            "High_Strain_Centroid_X":
                centroid_x,

            "High_Strain_Centroid_Y":
                centroid_y,

            "High_Strain_X_Min":
                x_min,

            "High_Strain_X_Max":
                x_max,

            "High_Strain_Y_Min":
                y_min,

            "High_Strain_Y_Max":
                y_max,

            "Validation_Image":
                output_image
        }
    )


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_csv = os.path.join(
    OUTPUT_DIR,
    "spatial_event_validation_summary.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 75)

print(
    "STAGE 15 COMPLETE"
)

print("=" * 75)

print()

print(
    f"Events successfully validated: "
    f"{len(results_df)}"
)

print()

print(
    "Summary:"
)

print(
    output_csv
)

print()

print(
    "Individual validation images:"
)

for path in results_df[
    "Validation_Image"
].tolist():

    print(
        path
    )

print()

print(
    "Review the five spatial validation images "
    "before finalizing event-detection logic."
)