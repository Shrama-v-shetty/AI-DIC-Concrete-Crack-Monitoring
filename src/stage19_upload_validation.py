import os
import re
import shutil
import tempfile

import pandas as pd
import streamlit as st


# ============================================================
# STAGE 19A - LOCAL FOLDER + EXCEL INPUT VALIDATION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "Stage19_Uploaded_Input"
)

EXCEL_OUTPUT_DIR = os.path.join(
    UPLOAD_DIR,
    "experimental_data"
)

os.makedirs(
    EXCEL_OUTPUT_DIR,
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
# SCAN LOCAL IMAGE FOLDER
# ============================================================

def scan_image_folder(folder_path):

    if not os.path.isdir(folder_path):
        return [], [], {}

    valid_extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    )

    valid_images = []
    invalid_images = []
    frame_map = {}

    for name in os.listdir(folder_path):

        full_path = os.path.join(
            folder_path,
            name
        )

        if not os.path.isfile(full_path):
            continue

        extension = os.path.splitext(
            name
        )[1].lower()

        if extension not in valid_extensions:

            invalid_images.append(
                (
                    name,
                    "Unsupported image format"
                )
            )

            continue

        frame = extract_frame_number(
            name
        )

        if frame is None:

            invalid_images.append(
                (
                    name,
                    "Could not determine frame number"
                )
            )

            continue

        # Avoid silently accepting duplicate frame numbers.
        if frame in frame_map:

            invalid_images.append(
                (
                    name,
                    f"Duplicate frame number: {frame}"
                )
            )

            continue

        frame_map[
            frame
        ] = full_path

        valid_images.append(
            full_path
        )

    analysis_frames = sorted(
        [
            frame
            for frame in frame_map
            if frame != 0
        ]
    )

    return (
        valid_images,
        invalid_images,
        frame_map
    ), analysis_frames


# ============================================================
# FIND MISSING FRAME RANGES
# ============================================================

def find_missing_ranges(frames):

    if not frames:
        return []

    frame_set = set(frames)

    start = min(frames)
    end = max(frames)

    missing_ranges = []

    missing_start = None

    for number in range(
        start,
        end + 1
    ):

        if number not in frame_set:

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
                end
            )
        )

    return missing_ranges


# ============================================================
# EXCEL VALIDATION
# ============================================================

def validate_excel(uploaded_excel):

    required_columns = [
        "Image Number",
        "Time_0",
        "Load ch 2 kN",
        "Displacement ch 7 µm"
    ]

    try:

        excel = pd.read_excel(
            uploaded_excel
        )

    except Exception as error:

        return (
            None,
            False,
            f"Could not read Excel file: {error}"
        )

    missing = [
        column
        for column in required_columns
        if column not in excel.columns
    ]

    if missing:

        return (
            excel,
            False,
            "Missing required columns: "
            + ", ".join(missing)
        )

    excel[
        "Image Number"
    ] = pd.to_numeric(
        excel[
            "Image Number"
        ],
        errors="coerce"
    )

    valid_numbers = excel[
        "Image Number"
    ].dropna()

    if valid_numbers.empty:

        return (
            excel,
            False,
            "No valid Image Number values found."
        )

    return (
        excel,
        True,
        "Excel validation successful."
    )


# ============================================================
# COPY EXCEL LOCALLY
# ============================================================

def save_excel(uploaded_excel):

    destination = os.path.join(
        EXCEL_OUTPUT_DIR,
        os.path.basename(
            uploaded_excel.name
        )
    )

    with open(
        destination,
        "wb"
    ) as file:

        file.write(
            uploaded_excel.getbuffer()
        )

    return destination


# ============================================================
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Digital Image Correlation based concrete "
    "deformation and safety monitoring"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Experiment Input"
)

st.sidebar.write(
    "Use a local image-folder path and upload "
    "the experimental Excel file."
)


# ============================================================
# IMAGE FOLDER PATH
# ============================================================

st.sidebar.subheader(
    "1. Image Dataset Folder"
)

folder_path = st.sidebar.text_input(
    "Local folder path",
    placeholder=(
        r"C:\Experiment\BGSL"
    ),
    help=(
        "Enter the path of the folder containing "
        "the experiment TIFF images."
    )
)

st.sidebar.caption(
    "The application reads images directly from "
    "this folder. Images are not uploaded through "
    "the browser."
)


# ============================================================
# EXCEL UPLOAD
# ============================================================

st.sidebar.subheader(
    "2. Experimental Excel"
)

uploaded_excel = st.sidebar.file_uploader(
    "Upload Excel file",
    type=[
        "xlsx",
        "xls"
    ],
    accept_multiple_files=False
)


# ============================================================
# MAIN INPUT DISPLAY
# ============================================================

st.subheader(
    "Experiment Input"
)

col1, col2 = st.columns(
    [2, 1]
)

with col1:

    if folder_path:

        if os.path.isdir(folder_path):

            st.success(
                "Image dataset folder found."
            )

            st.code(
                folder_path
            )

        else:

            st.error(
                "The entered folder does not exist."
            )

    else:

        st.info(
            "Enter the local image dataset folder path."
        )


with col2:

    if uploaded_excel is not None:

        st.success(
            f"Excel selected: "
            f"{uploaded_excel.name}"
        )

    else:

        st.info(
            "Upload the experimental Excel file."
        )


# ============================================================
# SCAN IMAGE DATASET
# ============================================================

image_scan = None
analysis_frames = []
valid_images = []
invalid_images = []
frame_map = {}

if folder_path and os.path.isdir(
    folder_path
):

    image_scan, analysis_frames = (
        scan_image_folder(
            folder_path
        )
    )

    (
        valid_images,
        invalid_images,
        frame_map
    ) = image_scan

    missing_ranges = find_missing_ranges(
        analysis_frames
    )

    st.divider()

    st.subheader(
        "Image Dataset Validation"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Image files",
        f"{len(valid_images):,}"
    )

    m2.metric(
        "Usable frames",
        f"{len(analysis_frames):,}"
    )

    m3.metric(
        "First frame",
        str(
            analysis_frames[0]
        )
        if analysis_frames
        else "-"
    )

    m4.metric(
        "Last frame",
        str(
            analysis_frames[-1]
        )
        if analysis_frames
        else "-"
    )


    if analysis_frames:

        st.write(
            "**Frame 0:** ignored automatically."
        )


    # --------------------------------------------------------
    # INVALID FILES
    # --------------------------------------------------------

    if invalid_images:

        st.warning(
            f"{len(invalid_images)} file(s) will "
            "not be used."
        )

        invalid_df = pd.DataFrame(
            invalid_images,
            columns=[
                "Filename",
                "Reason"
            ]
        )

        st.dataframe(
            invalid_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # MISSING RANGES
    # --------------------------------------------------------

    if missing_ranges:

        st.warning(
            f"{len(missing_ranges)} missing frame "
            "range(s) detected."
        )

        missing_df = pd.DataFrame(
            missing_ranges,
            columns=[
                "Missing_From",
                "Missing_To"
            ]
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Missing frames are preserved as gaps. "
            "They will not be treated as consecutive "
            "during sequential DIC."
        )

    else:

        st.success(
            "No missing frame ranges detected."
        )


# ============================================================
# EXCEL VALIDATION
# ============================================================

excel_data = None
excel_valid = False
excel_message = ""

if uploaded_excel is not None:

    (
        excel_data,
        excel_valid,
        excel_message
    ) = validate_excel(
        uploaded_excel
    )

    st.divider()

    st.subheader(
        "Experimental Excel Validation"
    )

    if excel_valid:

        st.success(
            excel_message
        )

        excel_frame_numbers = sorted(
            excel_data[
                "Image Number"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Excel rows",
            f"{len(excel_data):,}"
        )

        e2.metric(
            "Image-number entries",
            f"{len(excel_frame_numbers):,}"
        )

        e3.metric(
            "Excel columns",
            f"{len(excel_data.columns):,}"
        )

        st.write(
            "**Primary displacement:** "
            "Displacement ch 7 (µm)"
        )

        st.write(
            "**Primary load:** "
            "Load ch 2 (kN)"
        )

        st.write(
            "**Time:** Time_0"
        )

        with st.expander(
            "Preview Excel data"
        ):

            st.dataframe(
                excel_data.head(10),
                use_container_width=True,
                hide_index=True
            )

    else:

        st.error(
            excel_message
        )


# ============================================================
# CROSS-VALIDATION
# ============================================================

if (
    analysis_frames
    and excel_valid
):

    st.divider()

    st.subheader(
        "Image ↔ Excel Cross-Validation"
    )

    image_frame_set = set(
        analysis_frames
    )

    excel_frame_set = set(
        excel_frame_numbers
    )

    matched = (
        image_frame_set
        & excel_frame_set
    )

    image_without_excel = (
        image_frame_set
        - excel_frame_set
    )

    excel_without_image = (
        excel_frame_set
        - image_frame_set
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Image frames",
        f"{len(image_frame_set):,}"
    )

    c2.metric(
        "Matched frames",
        f"{len(matched):,}"
    )

    c3.metric(
        "Excel frames",
        f"{len(excel_frame_set):,}"
    )


    if image_without_excel:

        st.warning(
            f"{len(image_without_excel)} image "
            "frame(s) have no Excel entry."
        )

    else:

        st.success(
            "Every usable uploaded image frame "
            "has an Excel entry."
        )


    if excel_without_image:

        st.warning(
            f"{len(excel_without_image)} Excel "
            "frame(s) have no image."
        )


# ============================================================
# ANALYSIS READINESS
# ============================================================

st.divider()

st.subheader(
    "Analysis Preparation"
)

ready = (
    bool(folder_path)
    and os.path.isdir(folder_path)
    and len(analysis_frames) > 0
    and uploaded_excel is not None
    and excel_valid
)


if ready:

    st.success(
        "Input validation passed. "
        "The experiment is ready for analysis."
    )

    if st.button(
        "Prepare Experiment",
        type="primary"
    ):

        try:

            excel_saved = save_excel(
                uploaded_excel
            )

            # Store the selected folder path in a
            # small configuration file. This allows
            # later stages to use the same dataset
            # without copying the images.
            config_path = os.path.join(
                UPLOAD_DIR,
                "experiment_config.txt"
            )

            with open(
                config_path,
                "w",
                encoding="utf-8"
            ) as config:

                config.write(
                    f"IMAGE_DATASET={folder_path}\n"
                )

                config.write(
                    f"EXCEL_FILE={excel_saved}\n"
                )

                config.write(
                    f"FIRST_FRAME={analysis_frames[0]}\n"
                )

                config.write(
                    f"LAST_FRAME={analysis_frames[-1]}\n"
                )

                config.write(
                    f"USABLE_FRAMES={len(analysis_frames)}\n"
                )


            st.success(
                "Experiment prepared successfully."
            )

            st.write(
                f"**Images:** {len(analysis_frames):,}"
            )

            st.write(
                f"**Excel:** {uploaded_excel.name}"
            )

            st.write(
                f"**Configuration:** {config_path}"
            )

            st.info(
                "The images remain in their original "
                "folder; they are not copied or uploaded. "
                "The next stage will connect this validated "
                "input to the DIC processing engine."
            )

        except Exception as error:

            st.error(
                f"Could not prepare experiment: "
                f"{error}"
            )

else:

    st.warning(
        "Enter a valid image-folder path and upload "
        "a valid experimental Excel file."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Local mode: image files are read directly from "
    "the selected folder. Frame 0 is ignored. Missing "
    "frame ranges are preserved. Displacement channel 7 "
    "is the primary experimental displacement."
)
