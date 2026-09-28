import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STAGE 19B - ACTUAL FAST DIC PROCESSING
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage19B_DIC_Results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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
# SCAN IMAGE DIRECTORY
# ============================================================

def scan_images(folder):

    valid_extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    )

    frame_map = {}

    for name in os.listdir(folder):

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

        frame = extract_frame_number(
            name
        )

        if frame is None:
            continue

        if frame == 0:
            continue

        # Keep first image if duplicate frame exists
        if frame not in frame_map:

            frame_map[
                frame
            ] = path

    return frame_map


# ============================================================
# FAST DIC ENGINE
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
                cy - half_subset :
                cy + half_subset + 1,
                cx - half_subset :
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


            # Same rejection logic as validated Fast DIC
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


            center_score = (
                correlation[
                    center_y,
                    center_x
                ]
            )


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
# IMAGE LOADING
# ============================================================

def read_gray(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    return image


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
# PROCESS RANGE
# ============================================================

def process_range(
    frame_map,
    start_frame,
    end_frame,
    subset_size,
    search_size,
    step,
    progress_bar,
    status_text
):

    frames = [
        frame
        for frame in sorted(
            frame_map
        )
        if start_frame
        <= frame
        <= end_frame
    ]

    frames = sorted(
        frames
    )

    if len(frames) < 2:

        raise RuntimeError(
            "At least two available frames "
            "are required."
        )


    output_folder = os.path.join(
        OUTPUT_DIR,
        f"frames_{start_frame}_{end_frame}"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    summary_rows = []

    total_pairs = 0

    # Only process genuinely consecutive
    # available frames.
    for index in range(
        len(frames) - 1
    ):

        if (
            frames[index + 1]
            == frames[index] + 1
        ):

            total_pairs += 1


    completed = 0


    for index in range(
        len(frames) - 1
    ):

        frame1 = frames[index]
        frame2 = frames[index + 1]


        # Missing-frame gaps are never
        # treated as consecutive.
        if frame2 != frame1 + 1:

            continue


        status_text.write(
            f"Processing {frame1} → {frame2} "
            f"({completed + 1}/{total_pairs})"
        )


        start_time = time.time()


        img1 = read_gray(
            frame_map[frame1]
        )

        img2 = read_gray(
            frame_map[frame2]
        )


        results = track_pair_fast(
            img1,
            img2,
            subset_size=subset_size,
            search_size=search_size,
            step=step
        )


        summary = calculate_summary(
            results
        )


        processing_time = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # SAVE INDIVIDUAL DIC FIELD
        # ----------------------------------------------------

        pair_csv = os.path.join(
            output_folder,
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


        # ----------------------------------------------------
        # SUMMARY ROW
        # ----------------------------------------------------

        summary_rows.append(
            {
                "Frame_1": frame1,
                "Frame_2": frame2,

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


        completed += 1


        progress_bar.progress(
            completed / max(
                total_pairs,
                1
            )
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_csv = os.path.join(
        output_folder,
        "dic_summary.csv"
    )


    summary_df.to_csv(
        summary_csv,
        index=False
    )


    return (
        output_folder,
        summary_csv,
        summary_df
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Stage 19B — Local image dataset → Fast DIC"
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "Experiment Input"
)


folder_path = st.text_input(
    "Image dataset folder",
    placeholder=(
        r"C:\Experiment\BGSL"
    )
)


uploaded_excel = st.file_uploader(
    "Experimental Excel file",
    type=[
        "xlsx",
        "xls"
    ],
    accept_multiple_files=False
)


# ============================================================
# VALIDATION
# ============================================================

if folder_path:

    if not os.path.isdir(
        folder_path
    ):

        st.error(
            "The image dataset folder does not exist."
        )

        st.stop()

    frame_map = scan_images(
        folder_path
    )

else:

    frame_map = {}


if uploaded_excel is not None:

    try:

        excel = pd.read_excel(
            uploaded_excel
        )

        required_columns = [
            "Image Number",
            "Time_0",
            "Load ch 2 kN",
            "Displacement ch 7 µm"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in excel.columns
        ]

        if missing_columns:

            st.error(
                "Excel is missing required columns:\n"
                + ", ".join(
                    missing_columns
                )
            )

            excel_valid = False

        else:

            excel["Image Number"] = pd.to_numeric(
                excel["Image Number"],
                errors="coerce"
            )

            excel_valid = True

    except Exception as error:

        st.error(
            f"Could not read Excel: {error}"
        )

        excel_valid = False

else:

    excel = None
    excel_valid = False


# ============================================================
# DATASET SUMMARY
# ============================================================

if frame_map:

    frames = sorted(
        frame_map.keys()
    )

    st.subheader(
        "Dataset Summary"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Usable images",
        f"{len(frames):,}"
    )

    c2.metric(
        "First frame",
        str(frames[0])
    )

    c3.metric(
        "Last frame",
        str(frames[-1])
    )

    c4.metric(
        "Frame 0",
        "Ignored"
    )


    # --------------------------------------------------------
    # MISSING RANGES
    # --------------------------------------------------------

    missing_ranges = []

    missing_start = None

    for number in range(
        frames[0],
        frames[-1] + 1
    ):

        if number not in frame_map:

            if missing_start is None:
                missing_start = number

        else:

            if missing_start is not None:

                missing_ranges.append(
                    (
                        missing_start,
                        number - 1
                    )
                )

                missing_start = None


    if missing_start is not None:

        missing_ranges.append(
            (
                missing_start,
                frames[-1]
            )
        )


    if missing_ranges:

        st.warning(
            f"{len(missing_ranges)} missing "
            "frame range(s) detected."
        )

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

    else:

        st.success(
            "No missing frame ranges detected."
        )


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

st.divider()

st.subheader(
    "DIC Analysis Settings"
)


if frame_map:

    frames = sorted(
        frame_map.keys()
    )

    default_start = (
        1274
        if 1274 in frame_map
        else frames[0]
    )

    default_end = (
        1276
        if 1276 in frame_map
        else min(
            default_start + 2,
            frames[-1]
        )
    )

else:

    default_start = 1
    default_end = 2


col1, col2 = st.columns(2)


with col1:

    start_frame = st.number_input(
        "Start frame",
        min_value=1,
        value=default_start,
        step=1
    )


with col2:

    end_frame = st.number_input(
        "End frame",
        min_value=2,
        value=default_end,
        step=1
    )


col3, col4, col5 = st.columns(3)


with col3:

    subset_size = st.number_input(
        "Subset size",
        min_value=11,
        value=41,
        step=2
    )


with col4:

    search_size = st.number_input(
        "Search size",
        min_value=21,
        value=81,
        step=2
    )


with col5:

    step = st.number_input(
        "Grid step",
        min_value=1,
        value=10,
        step=1
    )


st.caption(
    "Default settings reproduce the validated Fast DIC "
    "configuration: subset 41, search 81, grid step 10."
)


# ============================================================
# VALIDATE RANGE
# ============================================================

range_valid = False

if frame_map:

    available = [
        frame
        for frame in frame_map
        if start_frame
        <= frame
        <= end_frame
    ]

    if len(available) >= 2:

        range_valid = True

        st.info(
            f"{len(available)} available image(s) "
            f"inside the selected range."
        )

    else:

        st.warning(
            "Select a range containing at least "
            "two available frames."
        )


# ============================================================
# START DIC
# ============================================================

st.divider()

ready = (
    bool(frame_map)
    and uploaded_excel is not None
    and excel_valid
    and range_valid
)


if ready:

    if st.button(
        "▶ Run Fast DIC",
        type="primary"
    ):

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            (
                output_folder,
                summary_csv,
                summary_df
            ) = process_range(
                frame_map,
                int(start_frame),
                int(end_frame),
                int(subset_size),
                int(search_size),
                int(step),
                progress,
                status
            )


            progress.progress(
                1.0
            )

            status.success(
                "DIC processing completed."
            )


            st.success(
                f"Processed {len(summary_df)} "
                "consecutive DIC pair(s)."
            )


            # ------------------------------------------------
            # RESULT SUMMARY
            # ------------------------------------------------

            if not summary_df.empty:

                st.subheader(
                    "DIC Results"
                )

                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Pairs",
                    f"{len(summary_df):,}"
                )

                r2.metric(
                    "Tracked points",
                    f"{int(summary_df['Tracked_Points'].mean()):,}"
                )

                r3.metric(
                    "Average ZNCC",
                    f"{summary_df['Average_ZNCC'].mean():.6f}"
                )

                r4.metric(
                    "Avg magnitude",
                    f"{summary_df['Average_Magnitude'].mean():.6f}"
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
                    output_folder
                )

                st.code(
                    summary_csv
                )

            else:

                st.warning(
                    "No consecutive frame pairs were found "
                    "inside the selected range."
                )


        except Exception as error:

            progress.empty()

            st.error(
                f"DIC processing failed:\n{error}"
            )

else:

    if not folder_path:

        st.info(
            "Enter the local image dataset folder."
        )

    elif uploaded_excel is None:

        st.info(
            "Upload the experimental Excel file."
        )

    elif not excel_valid:

        st.warning(
            "Fix the Excel validation issue first."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Frame 0 is ignored. Missing image ranges are never "
    "treated as consecutive. DIC uses ZNCC-based subset "
    "tracking with sub-pixel refinement."
)