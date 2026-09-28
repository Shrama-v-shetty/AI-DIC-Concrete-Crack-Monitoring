import os
import re
import json
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from streamlit_cropper import st_cropper

import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# STAGE 23 - FINAL INTEGRATED WORKFLOW
#
# Sequential DIC for every available frame
# + cumulative displacement
# + cumulative strain / heatmap
# + Excel synchronization
# ============================================================


BASE_DIR = (
    r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage23_Final_Integrated_Workflow"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

ROI_FILE = os.path.join(
    BASE_DIR,
    "Stage20_ROI_Selection",
    "roi.json"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="DIC Concrete Beam Monitoring System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "saved_roi" not in st.session_state:
    st.session_state.saved_roi = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "dic_summary_df" not in st.session_state:
    st.session_state.dic_summary_df = None

if "integrated_df" not in st.session_state:
    st.session_state.integrated_df = None

if "event_df" not in st.session_state:
    st.session_state.event_df = None

if "cumulative_field" not in st.session_state:
    st.session_state.cumulative_field = None

if "heatmap_path" not in st.session_state:
    st.session_state.heatmap_path = None


# ============================================================
# FRAME NUMBER
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
# SCAN IMAGES
# ============================================================

def scan_frames(folder):

    frame_map = {}

    extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    )

    try:
        filenames = os.listdir(folder)
    except Exception as error:

        raise RuntimeError(
            f"Could not read image folder: {error}"
        )

    for filename in filenames:

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        if not filename.lower().endswith(
            extensions
        ):
            continue

        frame = extract_frame_number(
            filename
        )

        if frame is None:
            continue

        # Frame 0 is never analyzed.
        if frame == 0:
            continue

        # Keep one image per frame.
        if frame not in frame_map:

            frame_map[
                frame
            ] = path

    return frame_map


# ============================================================
# FIND MISSING RANGES
# ============================================================

def find_missing_ranges(frames):

    if len(frames) < 2:
        return []

    frame_set = set(frames)

    missing = []

    start_missing = None

    for frame in range(
        frames[0],
        frames[-1] + 1
    ):

        if frame not in frame_set:

            if start_missing is None:
                start_missing = frame

        else:

            if start_missing is not None:

                missing.append(
                    (
                        start_missing,
                        frame - 1
                    )
                )

                start_missing = None

    if start_missing is not None:

        missing.append(
            (
                start_missing,
                frames[-1]
            )
        )

    return missing


# ============================================================
# LOAD ROI
# ============================================================

def load_roi():

    if not os.path.exists(
        ROI_FILE
    ):
        return None

    try:

        with open(
            ROI_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return None


# ============================================================
# SAVE ROI
# ============================================================

def save_roi(
    reference_frame,
    image_path,
    x,
    y,
    width,
    height
):

    roi = {
        "reference_frame": int(
            reference_frame
        ),

        "image_path": image_path,

        "x": int(x),
        "y": int(y),

        "width": int(width),
        "height": int(height)
    }

    with open(
        ROI_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            roi,
            file,
            indent=4
        )

    return roi


# ============================================================
# CROP IMAGE
# ============================================================

def crop_image(
    image,
    roi
):

    x = int(
        roi["x"]
    )

    y = int(
        roi["y"]
    )

    width = int(
        roi["width"]
    )

    height = int(
        roi["height"]
    )

    return image[
        y:y + height,
        x:x + width
    ]


# ============================================================
# SUBPIXEL
# ============================================================

def subpixel_offset(
    left,
    center,
    right
):

    denominator = (
        left
        - 2.0 * center
        + right
    )

    if abs(denominator) < 1e-10:
        return 0.0

    return 0.5 * (
        left - right
    ) / denominator


# ============================================================
# VALIDATED FAST DIC
# ============================================================

def track_pair_fast(
    img1,
    img2,
    subset_size=41,
    search_size=81,
    step=10
):

    half_subset = (
        subset_size // 2
    )

    half_search = (
        search_size // 2
    )

    margin = (
        half_subset
        + half_search
    )

    results = []

    for cy in range(
        margin,
        img1.shape[0] - margin,
        step
    ):

        for cx in range(
            margin,
            img1.shape[1] - margin,
            step
        ):

            template = img1[
                cy - half_subset:
                cy + half_subset + 1,

                cx - half_subset:
                cx + half_subset + 1
            ]

            if template.shape != (
                subset_size,
                subset_size
            ):
                continue

            if np.var(template) < 500:
                continue

            search_top = (
                cy - half_search
            )

            search_bottom = (
                cy + half_search + 1
            )

            search_left = (
                cx - half_search
            )

            search_right = (
                cx + half_search + 1
            )

            search_region = img2[
                search_top:search_bottom,
                search_left:search_right
            ]

            if search_region.shape != (
                search_size,
                search_size
            ):
                continue

            correlation = cv2.matchTemplate(
                search_region,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_score, _, max_location = (
                cv2.minMaxLoc(
                    correlation
                )
            )

            if np.isnan(max_score):
                continue

            if max_score < 0.98:
                continue

            match_x = (
                search_left
                + max_location[0]
            )

            match_y = (
                search_top
                + max_location[1]
            )

            best_dx = (
                match_x
                - (cx - half_subset)
            )

            best_dy = (
                match_y
                - (cy - half_subset)
            )

            if (
                abs(best_dx) > 5
                or abs(best_dy) > 5
            ):
                continue

            center_x = (
                max_location[0]
            )

            center_y = (
                max_location[1]
            )

            center_score = correlation[
                center_y,
                center_x
            ]

            left_score = center_score
            right_score = center_score
            up_score = center_score
            down_score = center_score

            if center_x > 0:

                left_score = correlation[
                    center_y,
                    center_x - 1
                ]

            if (
                center_x
                <
                correlation.shape[1] - 1
            ):

                right_score = correlation[
                    center_y,
                    center_x + 1
                ]

            if center_y > 0:

                up_score = correlation[
                    center_y - 1,
                    center_x
                ]

            if (
                center_y
                <
                correlation.shape[0] - 1
            ):

                down_score = correlation[
                    center_y + 1,
                    center_x
                ]

            best_dx += subpixel_offset(
                left_score,
                center_score,
                right_score
            )

            best_dy += subpixel_offset(
                up_score,
                center_score,
                down_score
            )

            results.append(
                [
                    cx,
                    cy,
                    best_dx,
                    best_dy,
                    float(max_score)
                ]
            )

    return results


# ============================================================
# DIC SUMMARY
# ============================================================

def summarize_dic(
    results
):

    if not results:

        return {
            "Tracked_Points": 0,
            "Average_ZNCC": np.nan,
            "Average_dx": np.nan,
            "Average_dy": np.nan,
            "Average_Magnitude": np.nan,
            "Maximum_Magnitude": np.nan
        }

    arr = np.asarray(
        results,
        dtype=float
    )

    dx = arr[:, 2]
    dy = arr[:, 3]
    zncc = arr[:, 4]

    magnitude = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    return {
        "Tracked_Points":
            len(arr),

        "Average_ZNCC":
            float(
                np.mean(zncc)
            ),

        "Average_dx":
            float(
                np.mean(dx)
            ),

        "Average_dy":
            float(
                np.mean(dy)
            ),

        "Average_Magnitude":
            float(
                np.mean(magnitude)
            ),

        "Maximum_Magnitude":
            float(
                np.max(magnitude)
            )
    }


# ============================================================
# BUILD EXPERIMENT LOOKUP
# ============================================================

def prepare_excel(
    excel
):

    required = [
        "Image Number",
        "Time_0",
        "Load ch 2 kN",
        "Displacement ch 7 µm"
    ]

    missing = [
        column
        for column in required
        if column not in excel.columns
    ]

    if missing:

        raise RuntimeError(
            "Excel is missing: "
            + ", ".join(missing)
        )

    excel = excel.copy()

    excel["Image Number"] = pd.to_numeric(
        excel["Image Number"],
        errors="coerce"
    )

    excel["Time_0"] = pd.to_numeric(
        excel["Time_0"],
        errors="coerce"
    )

    excel["Load ch 2 kN"] = pd.to_numeric(
        excel["Load ch 2 kN"],
        errors="coerce"
    )

    excel["Displacement ch 7 µm"] = pd.to_numeric(
        excel["Displacement ch 7 µm"],
        errors="coerce"
    )

    excel = excel[
        excel["Image Number"].notna()
    ].copy()

    excel["Image Number"] = (
        excel["Image Number"]
        .astype(int)
    )

    return (
        excel
        .drop_duplicates(
            subset=["Image Number"],
            keep="first"
        )
        .set_index(
            "Image Number"
        )
    )


# ============================================================
# CALCULATE STRAIN FROM CUMULATIVE FIELD
# ============================================================

def cumulative_strain(
    x,
    y,
    cumulative_dx,
    cumulative_dy,
    grid_step=10,
    sigma=2
):

    if len(x) < 50:
        return None

    gx_values = np.arange(
        np.floor(x.min()),
        np.ceil(x.max())
        + grid_step,
        grid_step
    )

    gy_values = np.arange(
        np.floor(y.min()),
        np.ceil(y.max())
        + grid_step,
        grid_step
    )

    gx, gy = np.meshgrid(
        gx_values,
        gy_values
    )

    gdx = griddata(
        (x, y),
        cumulative_dx,
        (gx, gy),
        method="linear"
    )

    gdy = griddata(
        (x, y),
        cumulative_dy,
        (gx, gy),
        method="linear"
    )

    for field, values in [
        (gdx, cumulative_dx),
        (gdy, cumulative_dy)
    ]:

        bad = ~np.isfinite(field)

        if np.any(bad):

            field[bad] = griddata(
                (x, y),
                values,
                (
                    gx[bad],
                    gy[bad]
                ),
                method="nearest"
            )

    gdx = gaussian_filter(
        gdx,
        sigma=sigma
    )

    gdy = gaussian_filter(
        gdy,
        sigma=sigma
    )

    du_dy, du_dx = np.gradient(
        gdx,
        grid_step,
        grid_step
    )

    dv_dy, dv_dx = np.gradient(
        gdy,
        grid_step,
        grid_step
    )

    exx = du_dx
    eyy = dv_dy

    exy = 0.5 * (
        du_dy
        + dv_dx
    )

    mean_strain = 0.5 * (
        exx
        + eyy
    )

    radius = np.sqrt(
        (
            (exx - eyy)
            * 0.5
        ) ** 2
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
# HEATMAP
# ============================================================

def create_heatmap(
    image,
    gx,
    gy,
    strain,
    output_path,
    title
):

    valid = np.isfinite(
        strain
    )

    if not np.any(valid):
        return

    # Do not let the fixture dominate visualization.
    stat_mask = ~(
        (gx >= 930)
        & (gx <= 1080)
        & (gy >= 450)
        & (gy <= 638)
    )

    valid = (
        np.isfinite(strain)
        & stat_mask
    )

    values = strain[
        valid
    ]

    if values.size == 0:
        return

    p99 = float(
        np.percentile(
            values,
            99
        )
    )

    if p99 <= 0:
        p99 = float(
            np.max(values)
        )

    if p99 <= 0:
        p99 = 1e-6


    cmap = LinearSegmentedColormap.from_list(
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


    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.imshow(
        strain,
        extent=[
            gx.min(),
            gx.max(),
            gy.max(),
            gy.min()
        ],
        cmap=cmap,
        alpha=0.58,
        vmin=0,
        vmax=p99
    )

    plt.colorbar(
        label="Principal strain"
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Image X (pixels)"
    )

    plt.ylabel(
        "Image Y (pixels)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# INTEGRATE DIC WITH EXCEL
# ============================================================

def integrate_data(
    dic_df,
    excel_lookup
):

    rows = []

    for _, row in dic_df.iterrows():

        frame1 = int(
            row["Frame_1"]
        )

        frame2 = int(
            row["Frame_2"]
        )

        if (
            frame1 not in excel_lookup.index
            or
            frame2 not in excel_lookup.index
        ):
            continue

        e1 = excel_lookup.loc[
            frame1
        ]

        e2 = excel_lookup.loc[
            frame2
        ]

        rows.append(
            {
                "Frame_1":
                    frame1,

                "Frame_2":
                    frame2,

                "Time_1":
                    e1["Time_0"],

                "Time_2":
                    e2["Time_0"],

                "Time_Delta_sec":
                    e2["Time_0"]
                    - e1["Time_0"],

                "Load_1_kN":
                    e1["Load ch 2 kN"],

                "Load_2_kN":
                    e2["Load ch 2 kN"],

                "Load_Delta_kN":
                    e2["Load ch 2 kN"]
                    - e1["Load ch 2 kN"],

                "Displacement_ch7_1_um":
                    e1[
                        "Displacement ch 7 µm"
                    ],

                "Displacement_ch7_2_um":
                    e2[
                        "Displacement ch 7 µm"
                    ],

                "Displacement_ch7_Delta_um":
                    e2[
                        "Displacement ch 7 µm"
                    ]
                    -
                    e1[
                        "Displacement ch 7 µm"
                    ],

                "Tracked_Points":
                    row["Tracked_Points"],

                "DIC_Average_ZNCC":
                    row["Average_ZNCC"],

                "DIC_Average_dx":
                    row["Average_dx"],

                "DIC_Average_dy":
                    row["Average_dy"],

                "DIC_Average_Magnitude":
                    row[
                        "Average_Magnitude"
                    ],

                "DIC_Maximum_Magnitude":
                    row[
                        "Maximum_Magnitude"
                    ],

                "DIC_Processing_Time_sec":
                    row[
                        "Processing_Time_sec"
                    ]
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# UI HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Sequential DIC + cumulative deformation "
    "+ experimental response"
)

st.divider()


# ============================================================
# INPUTS
# ============================================================

st.sidebar.header(
    "Experiment Input"
)

folder_path = st.sidebar.text_input(
    "Image dataset folder",
    value=(
        r"C:\Users\shrama\Desktop\Dam updated dataset"
        r"\Dataset\BGSL"
    )
)

uploaded_excel = st.sidebar.file_uploader(
    "Experimental Excel",
    type=[
        "xlsx",
        "xls"
    ]
)


# ============================================================
# VALIDATE FOLDER
# ============================================================

if not os.path.isdir(
    folder_path
):

    st.error(
        "Image dataset folder does not exist."
    )

    st.stop()


frame_map = scan_frames(
    folder_path
)

frames = sorted(
    frame_map.keys()
)


if len(frames) < 2:

    st.error(
        "At least two usable images are required."
    )

    st.stop()


# ============================================================
# DATASET STATUS
# ============================================================

st.subheader(
    "Dataset"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Usable images",
    f"{len(frames):,}"
)

c2.metric(
    "First frame",
    frames[0]
)

c3.metric(
    "Last frame",
    frames[-1]
)

c4.metric(
    "Frame 0",
    "Ignored"
)


missing_ranges = find_missing_ranges(
    frames
)

if missing_ranges:

    st.warning(
        f"{len(missing_ranges)} missing frame ranges detected."
    )

    with st.expander(
        "View missing ranges"
    ):

        st.dataframe(
            pd.DataFrame(
                missing_ranges,
                columns=[
                    "Missing_From",
                    "Missing_To"
                ]
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# LOAD SAVED ROI
# ============================================================

saved_roi = st.session_state.saved_roi

if saved_roi is None:

    saved_roi = load_roi()

    if saved_roi is not None:

        st.session_state.saved_roi = (
            saved_roi
        )


# ============================================================
# ROI SELECTION
# ============================================================

st.divider()

st.header(
    "1. Select Reference Image and ROI"
)


default_reference = (
    1
    if 1 in frame_map
    else frames[0]
)


reference_frame = st.selectbox(
    "Reference frame",
    frames,
    index=frames.index(
        default_reference
    )
)


reference_path = frame_map[
    reference_frame
]


reference_cv = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)


if reference_cv is None:

    st.error(
        "Could not read reference image."
    )

    st.stop()


reference_pil = Image.fromarray(
    reference_cv
)


# ------------------------------------------------------------
# Warn if saved ROI belongs to another reference
# ------------------------------------------------------------

if (
    saved_roi is not None
    and
    saved_roi["reference_frame"]
    != reference_frame
):

    st.warning(
        "The saved ROI belongs to reference frame "
        f"{saved_roi['reference_frame']}. "
        "Select that frame or draw and save a new ROI."
    )


st.write(
    f"Reference image: "
    f"**{os.path.basename(reference_path)}**"
)


box = st_cropper(
    reference_pil,

    realtime_update=True,

    box_color="#ff0000",

    return_type="box",

    aspect_ratio=None,

    should_resize_image=True,

    stroke_width=3,

    key=(
        f"stage23_roi_{reference_frame}"
    )
)


if box is not None:

    roi_x = int(
        box["left"]
    )

    roi_y = int(
        box["top"]
    )

    roi_width = int(
        box["width"]
    )

    roi_height = int(
        box["height"]
    )


    roi_valid = (
        roi_width > 10
        and roi_height > 10
        and roi_x >= 0
        and roi_y >= 0
        and roi_x + roi_width
        <= reference_cv.shape[1]
        and roi_y + roi_height
        <= reference_cv.shape[0]
    )


    if roi_valid:

        current_roi = {
            "reference_frame":
                int(reference_frame),

            "image_path":
                reference_path,

            "x":
                roi_x,

            "y":
                roi_y,

            "width":
                roi_width,

            "height":
                roi_height
        }


        st.session_state.saved_roi = (
            current_roi
        )


        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "X",
            roi_x
        )

        r2.metric(
            "Y",
            roi_y
        )

        r3.metric(
            "Width",
            roi_width
        )

        r4.metric(
            "Height",
            roi_height
        )


        if st.button(
            "✓ Save ROI",
            type="primary"
        ):

            save_roi(
                reference_frame,
                reference_path,
                roi_x,
                roi_y,
                roi_width,
                roi_height
            )

            st.success(
                "ROI saved successfully."
            )

            st.session_state.roi_saved = (
                True
            )


else:

    current_roi = saved_roi


# ============================================================
# DISPLAY CURRENT ROI
# ============================================================

saved_roi = (
    st.session_state.saved_roi
)


if saved_roi is None:

    st.warning(
        "Select and save an ROI before running analysis."
    )

else:

    st.success(
        "ROI ready for DIC."
    )

    st.write(
        f"Current ROI: "
        f"X={saved_roi['x']}, "
        f"Y={saved_roi['y']}, "
        f"W={saved_roi['width']}, "
        f"H={saved_roi['height']}"
    )


# ============================================================
# EXCEL
# ============================================================

st.divider()

st.header(
    "2. Experimental Data"
)


if uploaded_excel is None:

    st.warning(
        "Upload the experimental Excel file."
    )

    excel_lookup = None

else:

    try:

        excel_raw = pd.read_excel(
            uploaded_excel
        )

        excel_lookup = prepare_excel(
            excel_raw
        )

        st.success(
            "Experimental Excel validated."
        )

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Excel rows",
            f"{len(excel_raw):,}"
        )

        e2.metric(
            "Load",
            "Load ch 2"
        )

        e3.metric(
            "Displacement",
            "Ch7"
        )

    except Exception as error:

        excel_lookup = None

        st.error(
            f"Excel validation failed:\n{error}"
        )


# ============================================================
# ANALYSIS MODE
# ============================================================

st.divider()

st.header(
    "3. Analysis Range"
)


analysis_mode = st.radio(
    "Choose analysis mode",
    [
        "Test range",
        "Full available dataset"
    ],
    horizontal=True
)


if analysis_mode == "Test range":

    c1, c2 = st.columns(2)

    with c1:

        start_frame = st.number_input(
            "Start frame",
            min_value=1,
            value=(
                1274
                if 1274 in frame_map
                else frames[0]
            ),
            step=1
        )

    with c2:

        end_frame = st.number_input(
            "End frame",
            min_value=2,
            value=(
                1276
                if 1276 in frame_map
                else min(
                    frames[0] + 2,
                    frames[-1]
                )
            ),
            step=1
        )

else:

    start_frame = frames[0]
    end_frame = frames[-1]

    st.warning(
        "Full analysis processes every available consecutive "
        "image pair. Missing frame ranges are skipped. "
        "This can take a substantial amount of time."
    )


# ============================================================
# DIC SETTINGS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:

    subset_size = st.number_input(
        "Subset size",
        min_value=11,
        value=41,
        step=2
    )

with c2:

    search_size = st.number_input(
        "Search size",
        min_value=21,
        value=81,
        step=2
    )

with c3:

    grid_step = st.number_input(
        "Grid step",
        min_value=1,
        value=10,
        step=1
    )


# ============================================================
# AVAILABLE ANALYSIS FRAMES
# ============================================================

analysis_frames = [
    frame
    for frame in frames
    if start_frame
    <= frame
    <= end_frame
]


consecutive_pairs = []

for i in range(
    len(analysis_frames) - 1
):

    frame1 = analysis_frames[i]
    frame2 = analysis_frames[i + 1]

    if frame2 == frame1 + 1:

        consecutive_pairs.append(
            (
                frame1,
                frame2
            )
        )


p1, p2 = st.columns(2)

p1.metric(
    "Frames in range",
    f"{len(analysis_frames):,}"
)

p2.metric(
    "Sequential DIC pairs",
    f"{len(consecutive_pairs):,}"
)


# ============================================================
# START ANALYSIS
# ============================================================

st.divider()

ready = (
    saved_roi is not None
    and excel_lookup is not None
    and len(consecutive_pairs) > 0
)


run_button = st.button(
    "▶ Start Complete DIC Analysis",
    type="primary",
    disabled=not ready
)


if run_button:

    progress = st.progress(
        0
    )

    status = st.empty()

    summary_rows = []

    cumulative_states = []

    cumulative_dx = None
    cumulative_dy = None

    base_x = None
    base_y = None


    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    result_dir = os.path.join(
        OUTPUT_DIR,
        f"analysis_{timestamp}"
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )


    try:

        # ====================================================
        # SEQUENTIAL DIC
        # ====================================================

        total_pairs = len(
            consecutive_pairs
        )


        for pair_number, (
            frame1,
            frame2
        ) in enumerate(
            consecutive_pairs,
            start=1
        ):

            status.write(
                f"Processing "
                f"{frame1} → {frame2} "
                f"({pair_number}/{total_pairs})"
            )


            start_time = time.time()


            # ------------------------------------------------
            # READ IMAGES
            # ------------------------------------------------

            img1_full = cv2.imread(
                frame_map[frame1],
                cv2.IMREAD_GRAYSCALE
            )

            img2_full = cv2.imread(
                frame_map[frame2],
                cv2.IMREAD_GRAYSCALE
            )


            if (
                img1_full is None
                or img2_full is None
            ):

                raise RuntimeError(
                    f"Could not read frames "
                    f"{frame1} and {frame2}."
                )


            # ------------------------------------------------
            # APPLY SAME ROI
            # ------------------------------------------------

            img1 = crop_image(
                img1_full,
                saved_roi
            )

            img2 = crop_image(
                img2_full,
                saved_roi
            )


            if img1.shape != img2.shape:

                raise RuntimeError(
                    f"ROI dimensions differ for "
                    f"{frame1} → {frame2}."
                )


            # ------------------------------------------------
            # RUN FAST DIC
            # ------------------------------------------------

            results = track_pair_fast(
                img1,
                img2,
                subset_size=int(
                    subset_size
                ),
                search_size=int(
                    search_size
                ),
                step=int(
                    grid_step
                )
            )


            dic_stats = summarize_dic(
                results
            )


            processing_time = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # SAVE PAIR DIC
            # ------------------------------------------------

            pair_path = os.path.join(
                result_dir,
                f"dic_{frame1}_{frame2}.csv"
            )


            pair_df = pd.DataFrame(
                results,
                columns=[
                    "X",
                    "Y",
                    "dx",
                    "dy",
                    "ZNCC"
                ]
            )


            pair_df.to_csv(
                pair_path,
                index=False
            )


            # ------------------------------------------------
            # CUMULATIVE FIELD
            # ------------------------------------------------

            if len(results) > 0:

                arr = np.asarray(
                    results,
                    dtype=float
                )

                x = arr[:, 0]
                y = arr[:, 1]

                dx = arr[:, 2]
                dy = arr[:, 3]


                if cumulative_dx is None:

                    # First pair defines the common
                    # tracked-point reference grid.
                    base_x = x.copy()
                    base_y = y.copy()

                    cumulative_dx = dx.copy()
                    cumulative_dy = dy.copy()

                else:

                    # Interpolate the incremental
                    # displacement onto the current
                    # deformed locations.
                    current_x = (
                        base_x
                        + cumulative_dx
                    )

                    current_y = (
                        base_y
                        + cumulative_dy
                    )

                    interp_dx = griddata(
                        (x, y),
                        dx,
                        (
                            current_x,
                            current_y
                        ),
                        method="linear"
                    )

                    interp_dy = griddata(
                        (x, y),
                        dy,
                        (
                            current_x,
                            current_y
                        ),
                        method="linear"
                    )


                    # Nearest fallback
                    bad_dx = ~np.isfinite(
                        interp_dx
                    )

                    bad_dy = ~np.isfinite(
                        interp_dy
                    )


                    if np.any(bad_dx):

                        interp_dx[bad_dx] = griddata(
                            (x, y),
                            dx,
                            (
                                current_x[bad_dx],
                                current_y[bad_dx]
                            ),
                            method="nearest"
                        )


                    if np.any(bad_dy):

                        interp_dy[bad_dy] = griddata(
                            (x, y),
                            dy,
                            (
                                current_x[bad_dy],
                                current_y[bad_dy]
                            ),
                            method="nearest"
                        )


                    cumulative_dx += interp_dx
                    cumulative_dy += interp_dy


                cumulative_magnitude = np.sqrt(
                    cumulative_dx ** 2
                    + cumulative_dy ** 2
                )


                cumulative_states.append(
                    {
                        "Frame":
                            frame2,

                        "X":
                            base_x.copy(),

                        "Y":
                            base_y.copy(),

                        "Cumulative_dx":
                            cumulative_dx.copy(),

                        "Cumulative_dy":
                            cumulative_dy.copy(),

                        "Cumulative_Magnitude":
                            cumulative_magnitude.copy()
                    }
                )


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            summary_rows.append(
                {
                    "Frame_1":
                        frame1,

                    "Frame_2":
                        frame2,

                    "Tracked_Points":
                        dic_stats[
                            "Tracked_Points"
                        ],

                    "Average_ZNCC":
                        dic_stats[
                            "Average_ZNCC"
                        ],

                    "Average_dx":
                        dic_stats[
                            "Average_dx"
                        ],

                    "Average_dy":
                        dic_stats[
                            "Average_dy"
                        ],

                    "Average_Magnitude":
                        dic_stats[
                            "Average_Magnitude"
                        ],

                    "Maximum_Magnitude":
                        dic_stats[
                            "Maximum_Magnitude"
                        ],

                    "Processing_Time_sec":
                        processing_time,

                    "DIC_CSV":
                        pair_path
                }
            )


            progress.progress(
                pair_number / total_pairs
            )


        # ====================================================
        # SAVE DIC SUMMARY
        # ====================================================

        dic_summary_df = pd.DataFrame(
            summary_rows
        )


        dic_summary_path = os.path.join(
            result_dir,
            "dic_summary.csv"
        )


        dic_summary_df.to_csv(
            dic_summary_path,
            index=False
        )


        # ====================================================
        # SAVE CUMULATIVE FINAL FIELD
        # ====================================================

        final_cumulative_df = None

        if cumulative_states:

            final_state = (
                cumulative_states[-1]
            )


            final_cumulative_df = pd.DataFrame(
                {
                    "X":
                        final_state["X"],

                    "Y":
                        final_state["Y"],

                    "Cumulative_dx":
                        final_state[
                            "Cumulative_dx"
                        ],

                    "Cumulative_dy":
                        final_state[
                            "Cumulative_dy"
                        ],

                    "Cumulative_Magnitude":
                        final_state[
                            "Cumulative_Magnitude"
                        ]
                }
            )


            final_cumulative_path = os.path.join(
                result_dir,
                "final_cumulative_displacement.csv"
            )


            final_cumulative_df.to_csv(
                final_cumulative_path,
                index=False
            )


        # ====================================================
        # EXPERIMENTAL INTEGRATION
        # ====================================================

        integrated_df = integrate_data(
            dic_summary_df,
            excel_lookup
        )


        integrated_path = os.path.join(
            result_dir,
            "integrated_dic_experimental.csv"
        )


        integrated_df.to_csv(
            integrated_path,
            index=False
        )


        # ====================================================
        # EVENT DETECTION
        # ====================================================

        event_df = pd.DataFrame()


        if not integrated_df.empty:

            load_threshold = float(
                np.percentile(
                    np.abs(
                        integrated_df[
                            "Load_Delta_kN"
                        ]
                    ),
                    95
                )
            )


            disp_threshold = float(
                np.percentile(
                    np.abs(
                        integrated_df[
                            "Displacement_ch7_Delta_um"
                        ]
                    ),
                    95
                )
            )


            dic_threshold = float(
                np.percentile(
                    integrated_df[
                        "DIC_Average_Magnitude"
                    ],
                    95
                )
            )


            integrated_df[
                "Load_Drop"
            ] = (
                integrated_df[
                    "Load_Delta_kN"
                ] < 0
            )


            integrated_df[
                "Displacement_Rise"
            ] = (
                integrated_df[
                    "Displacement_ch7_Delta_um"
                ] > 0
            )


            integrated_df[
                "Significant_Load"
            ] = (
                np.abs(
                    integrated_df[
                        "Load_Delta_kN"
                    ]
                )
                >= load_threshold
            )


            integrated_df[
                "Significant_Displacement"
            ] = (
                np.abs(
                    integrated_df[
                        "Displacement_ch7_Delta_um"
                    ]
                )
                >= disp_threshold
            )


            integrated_df[
                "Significant_DIC"
            ] = (
                integrated_df[
                    "DIC_Average_Magnitude"
                ]
                >= dic_threshold
            )


            integrated_df[
                "Mechanical_Transition"
            ] = (
                integrated_df[
                    "Significant_Load"
                ]
                &
                integrated_df[
                    "Significant_Displacement"
                ]
            )


            integrated_df[
                "Load_Drop_Displacement_Rise"
            ] = (
                integrated_df["Load_Drop"]
                &
                integrated_df[
                    "Displacement_Rise"
                ]
            )


            integrated_df[
                "Correlated_Event"
            ] = (
                integrated_df[
                    "Mechanical_Transition"
                ]
                &
                integrated_df[
                    "Significant_DIC"
                ]
            )


            event_df = integrated_df[
                integrated_df[
                    "Load_Drop_Displacement_Rise"
                ]
            ].copy()


            event_path = os.path.join(
                result_dir,
                "event_candidates.csv"
            )


            event_df.to_csv(
                event_path,
                index=False
            )


            integrated_df.to_csv(
                integrated_path,
                index=False
            )


        # ====================================================
        # FINAL HEATMAP
        # ====================================================

        heatmap_path = None


        if (
            final_cumulative_df is not None
            and not final_cumulative_df.empty
        ):

            strain_result = cumulative_strain(
                final_cumulative_df["X"]
                .to_numpy(),

                final_cumulative_df["Y"]
                .to_numpy(),

                final_cumulative_df[
                    "Cumulative_dx"
                ].to_numpy(),

                final_cumulative_df[
                    "Cumulative_dy"
                ].to_numpy(),

                grid_step=int(
                    grid_step
                ),

                sigma=2
            )


            if strain_result is not None:

                gx, gy, strain = (
                    strain_result
                )


                final_frame = (
                    int(
                        dic_summary_df.iloc[-1][
                            "Frame_2"
                        ]
                    )
                )


                final_image = cv2.imread(
                    frame_map[
                        final_frame
                    ],
                    cv2.IMREAD_GRAYSCALE
                )


                final_image = crop_image(
                    final_image,
                    saved_roi
                )


                heatmap_path = os.path.join(
                    result_dir,
                    (
                        f"cumulative_heatmap_"
                        f"{final_frame}.png"
                    )
                )


                create_heatmap(
                    final_image,
                    gx,
                    gy,
                    strain,
                    heatmap_path,
                    (
                        "Cumulative DIC Principal "
                        f"Strain — Frame {final_frame}"
                    )
                )


        # ====================================================
        # SAVE CHECKPOINT SUMMARY
        # ====================================================

        checkpoint_rows = []

        for state in cumulative_states:

            magnitude = state[
                "Cumulative_Magnitude"
            ]

            checkpoint_rows.append(
                {
                    "Frame":
                        state["Frame"],

                    "Average_Cumulative_Magnitude":
                        float(
                            np.mean(
                                magnitude
                            )
                        ),

                    "Maximum_Cumulative_Magnitude":
                        float(
                            np.max(
                                magnitude
                            )
                        )
                }
            )


        checkpoint_df = pd.DataFrame(
            checkpoint_rows
        )


        checkpoint_path = os.path.join(
            result_dir,
            "cumulative_displacement_progression.csv"
        )


        checkpoint_df.to_csv(
            checkpoint_path,
            index=False
        )


        # ====================================================
        # STORE
        # ====================================================

        st.session_state.analysis_done = True

        st.session_state.dic_summary_df = (
            dic_summary_df
        )

        st.session_state.integrated_df = (
            integrated_df
        )

        st.session_state.event_df = (
            event_df
        )

        st.session_state.cumulative_field = (
            final_cumulative_df
        )

        st.session_state.heatmap_path = (
            heatmap_path
        )

        st.session_state.result_dir = (
            result_dir
        )


        progress.progress(
            1.0
        )

        status.success(
            "Complete sequential DIC analysis finished."
        )


    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            f"Analysis failed:\n{error}"
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    dic_summary_df = (
        st.session_state.dic_summary_df
    )

    integrated_df = (
        st.session_state.integrated_df
    )

    event_df = (
        st.session_state.event_df
    )

    cumulative_field = (
        st.session_state.cumulative_field
    )

    heatmap_path = (
        st.session_state.heatmap_path
    )


    st.divider()

    st.header(
        "4. Final Analysis Results"
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "DIC pairs",
        f"{len(dic_summary_df):,}"
    )

    m2.metric(
        "Avg tracked points",
        f"{int(dic_summary_df['Tracked_Points'].mean()):,}"
    )

    m3.metric(
        "Avg ZNCC",
        f"{dic_summary_df['Average_ZNCC'].mean():.6f}"
    )

    if cumulative_field is not None:

        m4.metric(
            "Final max cumulative motion",
            f"{cumulative_field['Cumulative_Magnitude'].max():.4f} px"
        )

    else:

        m4.metric(
            "Final cumulative motion",
            "Not available"
        )


    # ========================================================
    # RESULT TABS
    # ========================================================

    tabs = st.tabs(
        [
            "Integrated Data",
            "DIC",
            "Graphs",
            "Events",
            "Heatmap"
        ]
    )


    # --------------------------------------------------------
    # INTEGRATED DATA
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "DIC + Experimental Data"
        )

        st.dataframe(
            integrated_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # DIC
    # --------------------------------------------------------

    with tabs[1]:

        st.subheader(
            "Sequential DIC Results"
        )

        st.dataframe(
            dic_summary_df,
            use_container_width=True,
            hide_index=True
        )


        if cumulative_field is not None:

            st.subheader(
                "Final Cumulative Displacement Field"
            )

            st.dataframe(
                cumulative_field.head(1000),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "The table is limited to the first 1,000 "
                "points for display; the complete field is "
                "saved in the analysis output."
            )


    # --------------------------------------------------------
    # GRAPHS
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader(
            "Load vs Time"
        )

        load_plot = excel_lookup[
            [
                "Time_0",
                "Load ch 2 kN"
            ]
        ].dropna()

        if not load_plot.empty:

            st.line_chart(
                load_plot.set_index(
                    "Time_0"
                )
            )


        st.subheader(
            "Displacement vs Time"
        )

        disp_plot = excel_lookup[
            [
                "Time_0",
                "Displacement ch 7 µm"
            ]
        ].dropna()

        if not disp_plot.empty:

            st.line_chart(
                disp_plot.set_index(
                    "Time_0"
                )
            )


        st.subheader(
            "Load vs Displacement"
        )

        ld_plot = excel_lookup[
            [
                "Displacement ch 7 µm",
                "Load ch 2 kN"
            ]
        ].dropna()

        if not ld_plot.empty:

            st.scatter_chart(
                ld_plot,
                x="Displacement ch 7 µm",
                y="Load ch 2 kN"
            )


        st.subheader(
            "Incremental DIC Deformation"
        )

        if not dic_summary_df.empty:

            dic_plot = dic_summary_df[
                [
                    "Frame_2",
                    "Average_Magnitude"
                ]
            ].set_index(
                "Frame_2"
            )

            st.line_chart(
                dic_plot
            )


    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader(
            "Mechanical / DIC Event Candidates"
        )

        if event_df.empty:

            st.info(
                "No load-drop + displacement-rise "
                "candidates were found in the selected range."
            )

        else:

            st.write(
                f"{len(event_df)} candidate interval(s) found."
            )

            event_columns = [
                "Frame_1",
                "Frame_2",
                "Time_1",
                "Time_2",
                "Load_Delta_kN",
                "Displacement_ch7_Delta_um",
                "DIC_Average_Magnitude",
                "DIC_Maximum_Magnitude",
                "DIC_Average_ZNCC"
            ]

            event_columns = [
                c
                for c in event_columns
                if c in event_df.columns
            ]

            st.dataframe(
                event_df[
                    event_columns
                ],
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    with tabs[4]:

        st.subheader(
            "Cumulative DIC Principal-Strain Heatmap"
        )

        if (
            heatmap_path is not None
            and
            os.path.exists(
                heatmap_path
            )
        ):

            st.image(
                Image.open(
                    heatmap_path
                ),
                use_container_width=True
            )

            st.caption(
                "This heatmap represents cumulative deformation "
                "relative to the beginning of the selected "
                "analysis sequence, rather than only the final "
                "single-frame increment."
            )

        else:

            st.info(
                "No cumulative heatmap was generated."
            )


    # ========================================================
    # OUTPUT LOCATION
    # ========================================================

    st.divider()

    st.subheader(
        "5. Saved Analysis"
    )

    result_dir = st.session_state.get(
        "result_dir",
        None
    )

    if result_dir:

        st.code(
            result_dir
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Every available consecutive frame pair in the selected "
    "range is analysed using sequential DIC. Missing frame "
    "ranges are preserved as gaps. Cumulative displacement "
    "is maintained across the sequence so that strain/heatmap "
    "visualization is not based solely on a tiny one-frame "
    "increment. High-strain regions indicate deformation and "
    "are not automatically classified as confirmed cracks."
)