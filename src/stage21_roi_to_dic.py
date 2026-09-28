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


# ============================================================
# STAGE 21 - ROI -> FAST DIC INTEGRATION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

ROI_DIR = os.path.join(
    BASE_DIR,
    "Stage20_ROI_Selection"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage21_ROI_DIC_Integration"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

ROI_JSON = os.path.join(
    ROI_DIR,
    "roi.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DIC Concrete Beam Monitoring",
    page_icon="📐",
    layout="wide"
)


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
# SCAN IMAGE FOLDER
# ============================================================

def scan_frames(folder):

    frame_map = {}

    valid_extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    )

    try:
        names = os.listdir(folder)
    except Exception:
        return frame_map

    for name in names:

        path = os.path.join(
            folder,
            name
        )

        if not os.path.isfile(path):
            continue

        if not name.lower().endswith(
            valid_extensions
        ):
            continue

        frame = extract_frame_number(name)

        if frame is None:
            continue

        if frame == 0:
            continue

        if frame not in frame_map:
            frame_map[frame] = path

    return frame_map


# ============================================================
# LOAD ROI
# ============================================================

def load_roi():

    if not os.path.exists(
        ROI_JSON
    ):
        return None

    try:

        with open(
            ROI_JSON,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return None


# ============================================================
# CROP IMAGE
# ============================================================

def crop_image(
    image,
    roi
):

    x = int(roi["x"])
    y = int(roi["y"])
    width = int(roi["width"])
    height = int(roi["height"])

    return image[
        y:y + height,
        x:x + width
    ]


# ============================================================
# SUBPIXEL REFINEMENT
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
# VALIDATED FAST DIC ENGINE
# ============================================================

def track_pair_fast(
    img1,
    img2,
    subset_size=41,
    search_size=81,
    step=10
):

    half_subset = subset_size // 2
    half_search = search_size // 2

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

            # Same rejection logic as Stage 9 / Stage 19B.
            if (
                abs(best_dx) > 5
                or abs(best_dy) > 5
            ):
                continue

            center_x = max_location[0]
            center_y = max_location[1]

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
                < correlation.shape[1] - 1
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
                < correlation.shape[0] - 1
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

def calculate_summary(results):

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
        "Tracked_Points": len(arr),

        "Average_ZNCC": float(
            np.mean(zncc)
        ),

        "Average_dx": float(
            np.mean(dx)
        ),

        "Average_dy": float(
            np.mean(dy)
        ),

        "Average_Magnitude": float(
            np.mean(magnitude)
        ),

        "Maximum_Magnitude": float(
            np.max(magnitude)
        )
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Stage 21 — Visual ROI connected to validated Fast DIC"
)

st.divider()


# ============================================================
# IMAGE FOLDER
# ============================================================

st.subheader(
    "1. Image Dataset"
)

folder_path = st.text_input(
    "Image dataset folder",
    value=(
        r"C:\Users\shrama\Desktop\Dam updated dataset"
        r"\Dataset\BGSL"
    )
)

if not os.path.isdir(
    folder_path
):

    st.error(
        "The image dataset folder does not exist."
    )

    st.stop()


# ============================================================
# SCAN FRAMES
# ============================================================

frame_map = scan_frames(
    folder_path
)

if len(frame_map) < 2:

    st.error(
        "At least two usable frames are required."
    )

    st.stop()


frames = sorted(
    frame_map.keys()
)


st.success(
    f"{len(frames):,} usable image frames found."
)


# ============================================================
# EXISTING ROI
# ============================================================

saved_roi = load_roi()


if saved_roi is None:

    st.warning(
        "No Stage 20 ROI has been saved yet."
    )

    st.info(
        "Run stage20_automatic_roi.py first, "
        "save the concrete beam ROI, then return here."
    )

    st.stop()


# ============================================================
# DISPLAY SAVED ROI
# ============================================================

st.subheader(
    "2. Saved ROI"
)

st.write(
    f"Reference frame: "
    f"**{saved_roi['reference_frame']}**"
)

st.write(
    f"ROI: "
    f"X={saved_roi['x']}, "
    f"Y={saved_roi['y']}, "
    f"W={saved_roi['width']}, "
    f"H={saved_roi['height']}"
)


# ============================================================
# TEST ROI
# ============================================================

st.subheader(
    "3. ROI Preview"
)

preview_frame = st.selectbox(
    "Preview frame",
    options=frames,

    index=(
        frames.index(
            saved_roi["reference_frame"]
        )
        if saved_roi["reference_frame"] in frames
        else 0
    )
)

preview_path = frame_map[
    preview_frame
]

preview_image = cv2.imread(
    preview_path,
    cv2.IMREAD_GRAYSCALE
)

if preview_image is None:

    st.error(
        "Could not read preview frame."
    )

    st.stop()


preview_cropped = crop_image(
    preview_image,
    saved_roi
)

st.image(
    preview_cropped,
    clamp=True,
    use_container_width=True,
    caption=(
        f"ROI applied to frame {preview_frame}"
    )
)


# ============================================================
# DIC SETTINGS
# ============================================================

st.divider()

st.subheader(
    "4. DIC Settings"
)

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

    step = st.number_input(
        "Grid step",
        min_value=1,
        value=10,
        step=1
    )

st.caption(
    "These defaults reproduce the validated Stage 9 / "
    "Stage 19B Fast DIC configuration."
)


# ============================================================
# TEST RANGE
# ============================================================

st.divider()

st.subheader(
    "5. DIC Test Range"
)

col1, col2 = st.columns(2)

with col1:

    default_start = (
        1274
        if 1274 in frame_map
        else frames[0]
    )

    start_frame = st.number_input(
        "Start frame",
        min_value=1,
        value=default_start,
        step=1
    )

with col2:

    default_end = (
        1276
        if 1276 in frame_map
        else min(
            default_start + 2,
            frames[-1]
        )
    )

    end_frame = st.number_input(
        "End frame",
        min_value=2,
        value=default_end,
        step=1
    )


selected_frames = [
    frame
    for frame in frames
    if start_frame
    <= frame
    <= end_frame
]


consecutive_pairs = []

for i in range(
    len(selected_frames) - 1
):

    frame1 = selected_frames[i]
    frame2 = selected_frames[i + 1]

    if frame2 == frame1 + 1:

        consecutive_pairs.append(
            (
                frame1,
                frame2
            )
        )


st.write(
    f"Available frames in range: "
    f"**{len(selected_frames)}**"
)

st.write(
    f"Consecutive DIC pairs: "
    f"**{len(consecutive_pairs)}**"
)


if len(consecutive_pairs) == 0:

    st.warning(
        "No consecutive image pairs exist in "
        "the selected range."
    )


# ============================================================
# RUN DIC
# ============================================================

st.divider()

if st.button(
    "▶ Run ROI-Based Fast DIC",
    type="primary",
    disabled=(
        len(consecutive_pairs) == 0
    )
):

    progress = st.progress(
        0
    )

    status = st.empty()

    summary_rows = []

    total = len(
        consecutive_pairs
    )


    try:

        # ----------------------------------------------------
        # PROCESS EACH PAIR
        # ----------------------------------------------------

        for pair_index, (
            frame1,
            frame2
        ) in enumerate(
            consecutive_pairs,
            start=1
        ):

            status.write(
                f"Processing "
                f"{frame1} → {frame2} "
                f"({pair_index}/{total})"
            )

            start_time = time.time()


            # ------------------------------------------------
            # READ ORIGINAL IMAGES
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
                    f"{frame1} or {frame2}."
                )


            # ------------------------------------------------
            # APPLY SAVED ROI
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
                    f"ROI images have different shapes "
                    f"for {frame1} -> {frame2}."
                )


            # ------------------------------------------------
            # RUN VALIDATED FAST DIC
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
                    step
                )
            )


            summary = calculate_summary(
                results
            )


            processing_time = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # SAVE POINT FIELD
            # ------------------------------------------------

            pair_csv = os.path.join(
                OUTPUT_DIR,
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
                pair_csv,
                index=False
            )


            # ------------------------------------------------
            # SAVE SUMMARY
            # ------------------------------------------------

            summary_rows.append(
                {
                    "Frame_1":
                        frame1,

                    "Frame_2":
                        frame2,

                    "Tracked_Points":
                        summary[
                            "Tracked_Points"
                        ],

                    "Average_ZNCC":
                        summary[
                            "Average_ZNCC"
                        ],

                    "Average_dx":
                        summary[
                            "Average_dx"
                        ],

                    "Average_dy":
                        summary[
                            "Average_dy"
                        ],

                    "Average_Magnitude":
                        summary[
                            "Average_Magnitude"
                        ],

                    "Maximum_Magnitude":
                        summary[
                            "Maximum_Magnitude"
                        ],

                    "Processing_Time_sec":
                        processing_time
                }
            )


            # ------------------------------------------------
            # UPDATE PROGRESS
            # ------------------------------------------------

            progress.progress(
                pair_index / total
            )


        # ----------------------------------------------------
        # SAVE SUMMARY CSV
        # ----------------------------------------------------

        summary_df = pd.DataFrame(
            summary_rows
        )

        summary_csv = os.path.join(
            OUTPUT_DIR,
            "roi_dic_summary.csv"
        )

        summary_df.to_csv(
            summary_csv,
            index=False
        )


        # ----------------------------------------------------
        # COMPLETION
        # ----------------------------------------------------

        progress.progress(
            1.0
        )

        status.success(
            "ROI-based Fast DIC completed successfully."
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.success(
            f"Processed "
            f"{len(summary_df)} DIC pair(s)."
        )


        if not summary_df.empty:

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "Pairs",
                f"{len(summary_df):,}"
            )

            r2.metric(
                "Avg tracked points",
                f"{int(summary_df['Tracked_Points'].mean()):,}"
            )

            r3.metric(
                "Average ZNCC",
                f"{summary_df['Average_ZNCC'].mean():.6f}"
            )

            r4.metric(
                "Avg DIC magnitude",
                f"{summary_df['Average_Magnitude'].mean():.6f}"
            )


            st.subheader(
                "DIC Results"
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


            st.write(
                "**Saved results:**"
            )

            st.code(
                OUTPUT_DIR
            )

            st.code(
                summary_csv
            )


    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            f"DIC processing failed:\n{error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "The selected ROI is applied internally to each image. "
    "The original image dataset is never modified. "
    "The DIC engine uses the same ZNCC-based Fast DIC "
    "configuration validated against Stage 9."
)