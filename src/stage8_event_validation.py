import os
import re
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\shrama\Desktop\Dam updated dataset"

EXCEL_FILE = os.path.join(
    ROOT,
    "experimental_data",
    "combined_image_load_data_0_to_6600.xlsx"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "Dataset",
    "BGSL_ROI"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "Dataset",
    "Stage8_Event_Validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EVENTS TO VALIDATE
# ============================================================

EVENTS = [
    1853,
    3007,
    3432,
    6413
]


# ============================================================
# IMAGE WINDOW
# ============================================================

BEFORE_OFFSET = 20
AFTER_OFFSET = 20


# ============================================================
# PRIMARY DISPLACEMENT CHANNEL
# ============================================================

DISPLACEMENT_COLUMN = (
    "Displacement ch 7 µm"
)


# ============================================================
# FIND FRAME NUMBER
# ============================================================

def get_frame_number(filename):

    numbers = re.findall(
        r"\d+",
        os.path.basename(filename)
    )

    if len(numbers) < 2:
        return None

    # Actual frame number is the second-last number
    return int(numbers[-2])


# ============================================================
# INDEX AVAILABLE IMAGES
# ============================================================

print("==============================================")
print(" STAGE 8 - EVENT VALIDATION")
print("==============================================")

print("\nIndexing ROI images...")

image_files = glob.glob(
    os.path.join(
        IMAGE_DIR,
        "*.tif"
    )
)

frame_to_file = {}

for file in image_files:

    frame = get_frame_number(file)

    if frame is not None:

        frame_to_file[frame] = file


print(
    "Available image frames:",
    len(frame_to_file)
)

print(
    "First frame:",
    min(frame_to_file)
)

print(
    "Last frame:",
    max(frame_to_file)
)


# ============================================================
# LOAD EXCEL
# ============================================================

print("\nLoading experimental data...")

df = pd.read_excel(
    EXCEL_FILE
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = [
    "Image Number",
    "Time_0",
    "Load ch 2 kN",
    DISPLACEMENT_COLUMN
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.sort_values(
    "Image Number"
).reset_index(
    drop=True
)


# ============================================================
# INTERPOLATE ANALYSIS VALUES
# ============================================================

df[
    "Load_Analysis"
] = df[
    "Load ch 2 kN"
].interpolate(
    limit_direction="both"
)

df[
    "Displacement_Analysis"
] = df[
    DISPLACEMENT_COLUMN
].interpolate(
    limit_direction="both"
)


# ============================================================
# VALIDATE ONE EVENT
# ============================================================

def validate_event(event_frame):

    print("\n==============================================")
    print(
        f"VALIDATING EVENT: {event_frame}"
    )
    print("==============================================")

    # --------------------------------------------------------
    # Check event image
    # --------------------------------------------------------

    if event_frame not in frame_to_file:

        print(
            "Event image is not available."
        )

        return

    before_frame = event_frame - BEFORE_OFFSET
    after_frame = event_frame + AFTER_OFFSET

    # Find nearest available frames
    available = sorted(
        frame_to_file.keys()
    )

    before_candidates = [
        f for f in available
        if f <= before_frame
    ]

    after_candidates = [
        f for f in available
        if f >= after_frame
    ]

    if not before_candidates:

        print("No before image available.")
        return

    if not after_candidates:

        print("No after image available.")
        return

    actual_before = max(
        before_candidates
    )

    actual_after = min(
        after_candidates
    )

    print(
        "Before frame:",
        actual_before
    )

    print(
        "Event frame:",
        event_frame
    )

    print(
        "After frame:",
        actual_after
    )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    before_image = cv2.imread(
        frame_to_file[actual_before],
        cv2.IMREAD_GRAYSCALE
    )

    event_image = cv2.imread(
        frame_to_file[event_frame],
        cv2.IMREAD_GRAYSCALE
    )

    after_image = cv2.imread(
        frame_to_file[actual_after],
        cv2.IMREAD_GRAYSCALE
    )

    if (
        before_image is None
        or event_image is None
        or after_image is None
    ):

        print(
            "Could not read one or more images."
        )

        return

    # --------------------------------------------------------
    # Get Excel rows
    # --------------------------------------------------------

    window = df[
        (
            df["Image Number"]
            >= actual_before
        )
        &
        (
            df["Image Number"]
            <= actual_after
        )
    ].copy()

    event_row = df[
        df["Image Number"]
        == event_frame
    ]

    # --------------------------------------------------------
    # Event values
    # --------------------------------------------------------

    if len(event_row) > 0:

        row = event_row.iloc[0]

        event_time = row[
            "Time_0"
        ]

        event_load = row[
            "Load ch 2 kN"
        ]

        event_disp = row[
            DISPLACEMENT_COLUMN
        ]

    else:

        event_time = np.nan
        event_load = np.nan
        event_disp = np.nan

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 11)
    )

    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.1, 1]
    )

    # --------------------------------------------------------
    # IMAGE 1
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        grid[0, 0]
    )

    ax1.imshow(
        before_image,
        cmap="gray"
    )

    ax1.set_title(
        f"Before — Frame {actual_before}"
    )

    ax1.set_xlabel(
        "X (pixels)"
    )

    ax1.set_ylabel(
        "Y (pixels)"
    )

    # --------------------------------------------------------
    # IMAGE 2
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        grid[0, 1]
    )

    ax2.imshow(
        event_image,
        cmap="gray"
    )

    ax2.set_title(
        f"EVENT — Frame {event_frame}"
    )

    ax2.set_xlabel(
        "X (pixels)"
    )

    ax2.set_ylabel(
        "Y (pixels)"
    )

    # --------------------------------------------------------
    # IMAGE 3
    # --------------------------------------------------------

    ax3 = fig.add_subplot(
        grid[0, 2]
    )

    ax3.imshow(
        after_image,
        cmap="gray"
    )

    ax3.set_title(
        f"After — Frame {actual_after}"
    )

    ax3.set_xlabel(
        "X (pixels)"
    )

    ax3.set_ylabel(
        "Y (pixels)"
    )

    # --------------------------------------------------------
    # LOAD GRAPH
    # --------------------------------------------------------

    ax4 = fig.add_subplot(
        grid[1, 0:2]
    )

    ax4.plot(
        window["Time_0"],
        window["Load_Analysis"],
        linewidth=1.5
    )

    if not np.isnan(event_time):

        ax4.axvline(
            event_time,
            linestyle="--",
            linewidth=1.5
        )

    ax4.set_title(
        "Load around candidate event"
    )

    ax4.set_xlabel(
        "Time"
    )

    ax4.set_ylabel(
        "Load (kN)"
    )

    ax4.grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # DISPLACEMENT GRAPH
    # --------------------------------------------------------

    ax5 = fig.add_subplot(
        grid[1, 2]
    )

    ax5.plot(
        window["Time_0"],
        window["Displacement_Analysis"],
        linewidth=1.5
    )

    if not np.isnan(event_time):

        ax5.axvline(
            event_time,
            linestyle="--",
            linewidth=1.5
        )

    ax5.set_title(
        "Displacement around candidate"
    )

    ax5.set_xlabel(
        "Time"
    )

    ax5.set_ylabel(
        "Displacement (µm)"
    )

    ax5.grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"Event Validation — Frame {event_frame}\n"
            f"Time: {event_time:.3f} | "
            f"Load: {event_load:.3f} kN | "
            f"Displacement: {event_disp:.1f} µm"
        ),
        fontsize=15
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"event_validation_{event_frame}.png"
    )

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )

    # --------------------------------------------------------
    # Print event information
    # --------------------------------------------------------

    print("\nEVENT DATA")

    print(
        f"Time         : {event_time}"
    )

    print(
        f"Load         : {event_load}"
    )

    print(
        f"Displacement : {event_disp}"
    )


# ============================================================
# PROCESS EVENTS
# ============================================================

for event in EVENTS:

    validate_event(
        event
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print(" STAGE 8 COMPLETE")
print("==============================================")

print(
    "Validation images saved to:"
)

print(
    OUTPUT_DIR
)