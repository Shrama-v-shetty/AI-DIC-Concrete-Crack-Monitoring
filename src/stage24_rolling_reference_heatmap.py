import os
import re
import json
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ============================================================
# STAGE 24
# ROLLING-REFERENCE DIC HEATMAP
# ============================================================
#
# Sequential DIC remains the basis for frame-to-frame motion.
#
# For strain/heatmap:
#
#       Frame N-W  --->  Frame N
#
# This avoids accumulating thousands of incremental DIC
# errors into one long-term displacement field.
# ============================================================


BASE_DIR = (
    r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"
)

ROI_FILE = os.path.join(
    BASE_DIR,
    "Stage20_ROI_Selection",
    "roi.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage24_Rolling_Reference_Heatmap"
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
    page_icon="📊",
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
# SCAN IMAGES
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

        filenames = os.listdir(
            folder
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not read image folder: {error}"
        )


    for filename in filenames:

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(
            path
        ):
            continue

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        frame = extract_frame_number(
            filename
        )

        if frame is None:
            continue

        if frame == 0:
            continue

        if frame not in frame_map:

            frame_map[
                frame
            ] = path


    return frame_map


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

            return json.load(
                file
            )

    except Exception:

        return None


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
# SUBPIXEL OFFSET
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
# FAST DIC
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


            # Same texture criterion used in
            # the validated Stage 9 engine.
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


            if np.isnan(
                max_score
            ):

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


            # Important:
            # keep the validated rejection rule.
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
# STRAIN CALCULATION
# ============================================================

def calculate_strain(
    dic_results,
    grid_step=10,
    sigma=2
):

    if len(dic_results) < 50:

        raise RuntimeError(
            "Too few DIC points for strain calculation."
        )


    arr = np.asarray(
        dic_results,
        dtype=float
    )


    x = arr[:, 0]
    y = arr[:, 1]

    dx = arr[:, 2]
    dy = arr[:, 3]


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

        bad = ~np.isfinite(
            field
        )


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


    # Smooth the DIC field slightly before
    # calculating derivatives.
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
        +
        exy ** 2
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
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    image,
    gx,
    gy,
    strain,
    output_path,
    title
):

    # Mask the known loading/fixture region
    # when determining the display range.
    mask = ~(
        (gx >= 930)
        &
        (gx <= 1080)
        &
        (gy >= 450)
        &
        (gy <= 638)
    )


    valid = (
        np.isfinite(strain)
        & mask
    )


    values = strain[
        valid
    ]


    if values.size == 0:

        raise RuntimeError(
            "No valid strain values available "
            "for heatmap."
        )


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


    cmap = create_colormap()


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
        alpha=0.60,
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
# LOAD EXCEL
# ============================================================

def load_excel(
    uploaded_file
):

    if uploaded_file is None:

        return None


    excel = pd.read_excel(
        uploaded_file
    )


    required_columns = [
        "Image Number",
        "Time_0",
        "Load ch 2 kN",
        "Displacement ch 7 µm"
    ]


    missing = [
        c
        for c in required_columns
        if c not in excel.columns
    ]


    if missing:

        raise RuntimeError(
            "Missing Excel columns: "
            + ", ".join(missing)
        )


    for column in required_columns:

        excel[column] = pd.to_numeric(
            excel[column],
            errors="coerce"
        )


    excel = excel[
        excel["Image Number"].notna()
    ].copy()


    excel[
        "Image Number"
    ] = excel[
        "Image Number"
    ].astype(int)


    return (
        excel
        .drop_duplicates(
            subset=[
                "Image Number"
            ]
        )
        .set_index(
            "Image Number"
        )
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Stage 24 — Rolling-reference DIC strain heatmap"
)

st.divider()


# ============================================================
# INPUT
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


if len(frame_map) < 2:

    st.error(
        "At least two usable images are required."
    )

    st.stop()


frames = sorted(
    frame_map.keys()
)


# ============================================================
# ROI
# ============================================================

saved_roi = load_roi()


if saved_roi is None:

    st.error(
        "No saved ROI was found."
    )

    st.info(
        "Run Stage 20 first and save the concrete "
        "beam ROI."
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader(
    "Dataset"
)


d1, d2, d3 = st.columns(3)


d1.metric(
    "Usable images",
    f"{len(frames):,}"
)


d2.metric(
    "First frame",
    frames[0]
)


d3.metric(
    "Last frame",
    frames[-1]
)


# ============================================================
# EXCEL
# ============================================================

excel_lookup = None


if uploaded_excel is not None:

    try:

        excel_lookup = load_excel(
            uploaded_excel
        )

        st.success(
            "Experimental Excel loaded successfully."
        )

    except Exception as error:

        st.error(
            f"Excel error: {error}"
        )

        st.stop()


else:

    st.warning(
        "Upload the experimental Excel file "
        "for synchronized measurements."
    )


# ============================================================
# ROLLING WINDOW
# ============================================================

st.divider()

st.header(
    "1. Rolling Reference"
)


window_size = st.number_input(
    "Reference window (frames)",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)


st.caption(
    "For target frame N, DIC is calculated between "
    "the frame located approximately "
    f"{window_size} frames earlier and frame N."
)


# ============================================================
# TARGET FRAME
# ============================================================

st.subheader(
    "2. Select Target Frame"
)


default_target = (
    1374
    if 1374 in frame_map
    else frames[
        min(
            len(frames) - 1,
            int(window_size)
        )
    ]
)


target_frame = st.selectbox(
    "Target frame",
    frames,
    index=frames.index(
        default_target
    )
)


target_index = frames.index(
    target_frame
)


if target_index < window_size:

    st.warning(
        "The selected target frame does not have "
        "enough previous available frames for the "
        "selected rolling window."
    )

    st.stop()


reference_index = (
    target_index
    - int(window_size)
)


reference_frame = frames[
    reference_index
]


st.write(
    f"Rolling-reference pair: "
    f"**{reference_frame} → {target_frame}**"
)


# ============================================================
# DIC SETTINGS
# ============================================================

st.divider()

st.header(
    "3. DIC Settings"
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

    grid_step = st.number_input(
        "Grid step",
        min_value=1,
        value=10,
        step=1
    )


st.caption(
    "Defaults match the validated Fast DIC configuration."
)


# ============================================================
# RUN
# ============================================================

st.divider()

if st.button(
    "▶ Generate Rolling-Reference Heatmap",
    type="primary"
):

    progress = st.progress(
        0
    )

    status = st.empty()


    try:

        # ----------------------------------------------------
        # READ REFERENCE
        # ----------------------------------------------------

        status.write(
            f"Reading reference frame {reference_frame}..."
        )


        image1 = cv2.imread(
            frame_map[
                reference_frame
            ],
            cv2.IMREAD_GRAYSCALE
        )


        # ----------------------------------------------------
        # READ TARGET
        # ----------------------------------------------------

        status.write(
            f"Reading target frame {target_frame}..."
        )


        image2 = cv2.imread(
            frame_map[
                target_frame
            ],
            cv2.IMREAD_GRAYSCALE
        )


        if (
            image1 is None
            or image2 is None
        ):

            raise RuntimeError(
                "Could not read one of the selected images."
            )


        progress.progress(
            0.10
        )


        # ----------------------------------------------------
        # APPLY ROI
        # ----------------------------------------------------

        roi_image1 = crop_image(
            image1,
            saved_roi
        )


        roi_image2 = crop_image(
            image2,
            saved_roi
        )


        if roi_image1.shape != roi_image2.shape:

            raise RuntimeError(
                "ROI dimensions differ."
            )


        # ----------------------------------------------------
        # RUN DIC
        # ----------------------------------------------------

        status.write(
            "Running rolling-reference Fast DIC..."
        )


        dic_start = time.time()


        results = track_pair_fast(
            roi_image1,
            roi_image2,

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


        dic_time = (
            time.time()
            - dic_start
        )


        if len(results) < 50:

            raise RuntimeError(
                f"Only {len(results)} DIC points were tracked."
            )


        progress.progress(
            0.65
        )


        # ----------------------------------------------------
        # DIC SUMMARY
        # ----------------------------------------------------

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


        dic_stats = {
            "Reference_Frame":
                reference_frame,

            "Target_Frame":
                target_frame,

            "Tracked_Points":
                len(results),

            "Average_ZNCC":
                float(
                    np.mean(
                        zncc
                    )
                ),

            "Average_dx":
                float(
                    np.mean(
                        dx
                    )
                ),

            "Average_dy":
                float(
                    np.mean(
                        dy
                    )
                ),

            "Average_Magnitude":
                float(
                    np.mean(
                        magnitude
                    )
                ),

            "Maximum_Magnitude":
                float(
                    np.max(
                        magnitude
                    )
                ),

            "Processing_Time_sec":
                dic_time
        }


        # ----------------------------------------------------
        # SAVE DIC FIELD
        # ----------------------------------------------------

        pair_name = (
            f"{reference_frame}_"
            f"{target_frame}"
        )


        dic_csv = os.path.join(
            OUTPUT_DIR,
            f"dic_{pair_name}.csv"
        )


        pd.DataFrame(
            results,
            columns=[
                "X",
                "Y",
                "dx",
                "dy",
                "ZNCC"
            ]
        ).to_csv(
            dic_csv,
            index=False
        )


        # ----------------------------------------------------
        # STRAIN
        # ----------------------------------------------------

        status.write(
            "Calculating principal strain..."
        )


        gx, gy, strain = (
            calculate_strain(
                results,

                grid_step=int(
                    grid_step
                ),

                sigma=2
            )
        )


        progress.progress(
            0.85
        )


        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        heatmap_path = os.path.join(
            OUTPUT_DIR,
            (
                f"rolling_heatmap_"
                f"{reference_frame}_"
                f"{target_frame}.png"
            )
        )


        target_image = crop_image(
            image2,
            saved_roi
        )


        create_heatmap(
            target_image,
            gx,
            gy,
            strain,
            heatmap_path,
            (
                "Rolling-Reference DIC Principal "
                f"Strain — {reference_frame} → {target_frame}"
            )
        )


        # ----------------------------------------------------
        # STRAIN STATISTICS
        # ----------------------------------------------------

        statistic_mask = ~(
            (gx >= 930)
            &
            (gx <= 1080)
            &
            (gy >= 450)
            &
            (gy <= 638)
        )


        valid_strain = (
            np.isfinite(strain)
            & statistic_mask
        )


        strain_values = strain[
            valid_strain
        ]


        mean_strain = float(
            np.mean(
                strain_values
            )
        )


        p95_strain = float(
            np.percentile(
                strain_values,
                95
            )
        )


        p99_strain = float(
            np.percentile(
                strain_values,
                99
            )
        )


        max_strain = float(
            np.max(
                strain_values
            )
        )


        # ----------------------------------------------------
        # EXPERIMENTAL VALUES
        # ----------------------------------------------------

        exp_info = None


        if excel_lookup is not None:

            if (
                reference_frame
                in excel_lookup.index
                and
                target_frame
                in excel_lookup.index
            ):

                e1 = excel_lookup.loc[
                    reference_frame
                ]

                e2 = excel_lookup.loc[
                    target_frame
                ]


                exp_info = {
                    "Time_Start":
                        float(
                            e1["Time_0"]
                        ),

                    "Time_End":
                        float(
                            e2["Time_0"]
                        ),

                    "Load_Start_kN":
                        float(
                            e1[
                                "Load ch 2 kN"
                            ]
                        ),

                    "Load_End_kN":
                        float(
                            e2[
                                "Load ch 2 kN"
                            ]
                        ),

                    "Load_Delta_kN":
                        float(
                            e2[
                                "Load ch 2 kN"
                            ]
                            -
                            e1[
                                "Load ch 2 kN"
                            ]
                        ),

                    "Ch7_Start_um":
                        float(
                            e1[
                                "Displacement ch 7 µm"
                            ]
                        ),

                    "Ch7_End_um":
                        float(
                            e2[
                                "Displacement ch 7 µm"
                            ]
                        ),

                    "Ch7_Delta_um":
                        float(
                            e2[
                                "Displacement ch 7 µm"
                            ]
                            -
                            e1[
                                "Displacement ch 7 µm"
                            ]
                        )
                }


        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        summary_row = {
            **dic_stats,

            "Mean_Principal_Strain":
                mean_strain,

            "P95_Principal_Strain":
                p95_strain,

            "P99_Principal_Strain":
                p99_strain,

            "Max_Principal_Strain":
                max_strain
        }


        if exp_info is not None:

            summary_row.update(
                exp_info
            )


        summary_csv = os.path.join(
            OUTPUT_DIR,
            (
                f"rolling_summary_"
                f"{reference_frame}_"
                f"{target_frame}.csv"
            )
        )


        pd.DataFrame(
            [summary_row]
        ).to_csv(
            summary_csv,
            index=False
        )


        # ----------------------------------------------------
        # RESULT STATE
        # ----------------------------------------------------

        st.session_state.stage24_summary = (
            summary_row
        )

        st.session_state.stage24_heatmap = (
            heatmap_path
        )


        progress.progress(
            1.0
        )


        status.success(
            "Rolling-reference DIC heatmap generated successfully."
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.divider()

        st.header(
            "4. DIC Results"
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Tracked points",
            f"{dic_stats['Tracked_Points']:,}"
        )


        c2.metric(
            "Average ZNCC",
            f"{dic_stats['Average_ZNCC']:.6f}"
        )


        c3.metric(
            "Avg DIC magnitude",
            f"{dic_stats['Average_Magnitude']:.6f}"
        )


        c4.metric(
            "DIC processing time",
            f"{dic_time:.2f} sec"
        )


        st.write(
            f"**Rolling reference:** "
            f"{reference_frame} → {target_frame}"
        )


        # ----------------------------------------------------
        # STRAIN
        # ----------------------------------------------------

        st.subheader(
            "Principal Strain"
        )


        s1, s2, s3, s4 = st.columns(4)


        s1.metric(
            "Mean",
            f"{mean_strain:.6f}"
        )


        s2.metric(
            "P95",
            f"{p95_strain:.6f}"
        )


        s3.metric(
            "P99",
            f"{p99_strain:.6f}"
        )


        s4.metric(
            "Maximum",
            f"{max_strain:.6f}"
        )


        # ----------------------------------------------------
        # EXPERIMENTAL RESPONSE
        # ----------------------------------------------------

        if exp_info is not None:

            st.subheader(
                "Experimental Response"
            )


            e1, e2, e3 = st.columns(3)


            e1.metric(
                "Load change",
                f"{exp_info['Load_Delta_kN']:.3f} kN"
            )


            e2.metric(
                "Ch7 displacement change",
                f"{exp_info['Ch7_Delta_um']:.1f} µm"
            )


            e3.metric(
                "Time difference",
                f"{exp_info['Time_End'] - exp_info['Time_Start']:.3f} sec"
            )


        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        st.divider()

        st.header(
            "5. DIC Heatmap"
        )


        if os.path.exists(
            heatmap_path
        ):

            st.image(
                Image.open(
                    heatmap_path
                ),
                use_container_width=True
            )


        # ----------------------------------------------------
        # FILE OUTPUTS
        # ----------------------------------------------------

        st.subheader(
            "Saved Results"
        )


        st.code(
            dic_csv
        )

        st.code(
            summary_csv
        )

        st.code(
            heatmap_path
        )


    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            f"Rolling-reference analysis failed:\n{error}"
        )


# ============================================================
# PREVIOUS RESULT
# ============================================================

if (
    "stage24_heatmap"
    in st.session_state
):

    st.divider()

    st.caption(
        "The displayed heatmap is generated from the "
        "selected rolling-reference interval."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sequential DIC is used for frame-to-frame monitoring. "
    "Rolling-reference DIC is used for strain visualization "
    "to reduce long-term accumulation of incremental DIC drift. "
    "A localized high-strain region indicates deformation and "
    "is not automatically classified as a confirmed crack."
)