import os
import re
import json

import cv2
import streamlit as st

from PIL import Image
from streamlit_cropper import st_cropper


# ============================================================
# STAGE 20 - MOUSE-BASED ROI SELECTION
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage20_ROI_Selection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

ROI_JSON = os.path.join(
    OUTPUT_DIR,
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
    """
    Example:

        BGSL-2-1274_0.tif

    Numeric groups:
        2
        1274
        0

    Actual frame:
        1274
    """

    numbers = re.findall(
        r"\d+",
        os.path.basename(filename)
    )

    if len(numbers) >= 2:
        return int(numbers[-2])

    return None


# ============================================================
# SCAN IMAGE DATASET
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
        filenames = os.listdir(folder)

    except Exception as error:

        st.error(
            f"Could not read the image folder: {error}"
        )

        return frame_map


    for filename in filenames:

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
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

        # Frame 0 is never used
        if frame == 0:
            continue

        # Ignore duplicate frame numbers
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

    missing_ranges = []

    missing_start = None

    for frame in range(
        frames[0],
        frames[-1] + 1
    ):

        if frame not in frame_set:

            if missing_start is None:
                missing_start = frame

        else:

            if missing_start is not None:

                missing_ranges.append(
                    (
                        missing_start,
                        frame - 1
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

    return missing_ranges


# ============================================================
# SAVE ROI
# ============================================================

def save_roi(
    reference_frame,
    reference_path,
    box
):

    roi = {
        "reference_frame": int(
            reference_frame
        ),

        "image_path": reference_path,

        "x": int(box["left"]),
        "y": int(box["top"]),

        "width": int(box["width"]),
        "height": int(box["height"])
    }

    with open(
        ROI_JSON,
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
# LOAD SAVED ROI
# ============================================================

def load_saved_roi():

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
# CROP IMAGE WITH ROI
# ============================================================

def crop_with_roi(
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
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Stage 20 — Visual concrete beam ROI selection"
)

st.divider()


# ============================================================
# STEP 1 - DATASET
# ============================================================

st.subheader(
    "1. Select Image Dataset"
)

folder_path = st.text_input(
    "Image dataset folder",
    placeholder=r"C:\Experiment\BGSL"
)

if not folder_path:

    st.info(
        "Enter the local image dataset folder path."
    )

    st.stop()


if not os.path.isdir(
    folder_path
):

    st.error(
        "The specified folder does not exist."
    )

    st.stop()


# ============================================================
# SCAN DATASET
# ============================================================

frame_map = scan_frames(
    folder_path
)

if not frame_map:

    st.error(
        "No valid experiment images were found."
    )

    st.stop()


frames = sorted(
    frame_map.keys()
)


# ============================================================
# DATASET INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Usable images",
    f"{len(frames):,}"
)

col2.metric(
    "First frame",
    str(frames[0])
)

col3.metric(
    "Last frame",
    str(frames[-1])
)

st.caption(
    "Frame 0 is automatically ignored."
)


# ============================================================
# MISSING FRAMES
# ============================================================

missing_ranges = find_missing_ranges(
    frames
)

if missing_ranges:

    st.warning(
        f"{len(missing_ranges)} missing frame "
        "range(s) detected."
    )

    missing_df = []

    for start, end in missing_ranges:

        missing_df.append(
            {
                "Missing from": start,
                "Missing to": end,
                "Number of frames": end - start + 1
            }
        )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Missing ranges will remain gaps during DIC "
        "processing and will not be treated as consecutive."
    )


# ============================================================
# STEP 2 - REFERENCE IMAGE
# ============================================================

st.divider()

st.subheader(
    "2. Select Reference Image"
)

default_reference = (
    1
    if 1 in frames
    else frames[0]
)


reference_frame = st.selectbox(
    "Reference frame",
    options=frames,
    index=frames.index(
        default_reference
    )
)


reference_path = frame_map[
    reference_frame
]


reference_image_cv = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)

if reference_image_cv is None:

    st.error(
        "Could not read the selected reference image."
    )

    st.stop()


# Convert OpenCV image -> PIL
reference_image = Image.fromarray(
    reference_image_cv
)


st.write(
    f"Reference image: "
    f"**{os.path.basename(reference_path)}**"
)

st.write(
    f"Original image size: "
    f"**{reference_image.width} × "
    f"{reference_image.height} pixels**"
)


# ============================================================
# STEP 3 - DRAW ROI
# ============================================================

st.divider()

st.subheader(
    "3. Select Concrete Beam Region"
)

st.write(
    "Drag the rectangle directly over the image "
    "to select the concrete beam region."
)

st.info(
    "Do not type coordinates. Just drag the box with "
    "your mouse. Include the useful concrete surface "
    "and exclude unnecessary machine/background areas."
)


# ============================================================
# CROPPER
# ============================================================

box = st_cropper(
    reference_image,

    realtime_update=True,

    box_color="#ff0000",

    return_type="box",

    aspect_ratio=None,

    should_resize_image=True,

    stroke_width=3,

    key=f"roi_cropper_{reference_frame}"
)


# ============================================================
# SELECTED ROI
# ============================================================

if box is not None:

    x = int(
        box["left"]
    )

    y = int(
        box["top"]
    )

    width = int(
        box["width"]
    )

    height = int(
        box["height"]
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    roi_valid = (
        width > 10
        and height > 10
        and x >= 0
        and y >= 0
        and x + width
        <= reference_image.width
        and y + height
        <= reference_image.height
    )


    if roi_valid:

        st.divider()

        st.subheader(
            "Selected ROI"
        )


        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "X",
            x
        )

        r2.metric(
            "Y",
            y
        )

        r3.metric(
            "Width",
            width
        )

        r4.metric(
            "Height",
            height
        )


        st.success(
            "ROI selected successfully."
        )


        # ----------------------------------------------------
        # PREVIEW CROPPED IMAGE
        # ----------------------------------------------------

        cropped_preview = crop_with_roi(
            reference_image_cv,
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        )


        st.write(
            "ROI preview"
        )


        st.image(
            cropped_preview,
            use_container_width=True,
            clamp=True,
            caption=(
                "This is the concrete region that will "
                "be used by the DIC pipeline."
            )
        )


        # ----------------------------------------------------
        # CONFIRM BUTTON
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "4. Confirm ROI"
        )


        if st.button(
            "✓ Confirm and Save ROI",
            type="primary"
        ):

            selected_box = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }


            saved_roi = save_roi(
                reference_frame,
                reference_path,
                selected_box
            )


            st.session_state[
                "saved_roi"
            ] = saved_roi


            st.success(
                "ROI configuration saved successfully."
            )


            st.write(
                f"Saved to:"
            )

            st.code(
                ROI_JSON
            )


            st.json(
                saved_roi
            )


    else:

        st.warning(
            "The selected ROI is too small or outside "
            "the image boundaries."
        )


# ============================================================
# LOAD SAVED ROI
# ============================================================

saved_roi = st.session_state.get(
    "saved_roi",
    None
)


if saved_roi is None:

    saved_roi = load_saved_roi()


# ============================================================
# SAVED ROI INFORMATION
# ============================================================

if saved_roi is not None:

    st.divider()

    st.subheader(
        "Saved ROI"
    )

    st.write(
        f"Reference frame: "
        f"**{saved_roi['reference_frame']}**"
    )

    st.write(
        f"X = **{saved_roi['x']}**"
    )

    st.write(
        f"Y = **{saved_roi['y']}**"
    )

    st.write(
        f"Width = **{saved_roi['width']}**"
    )

    st.write(
        f"Height = **{saved_roi['height']}**"
    )

    st.success(
        "ROI is ready for the DIC processing pipeline."
    )


# ============================================================
# STEP 5 - TEST ROI ON OTHER FRAMES
# ============================================================

st.divider()

st.subheader(
    "5. Test Saved ROI on Other Frames"
)

if saved_roi is None:

    st.info(
        "Save the ROI first."
    )

else:

    default_test_frames = [
        frame
        for frame in [
            1,
            1274,
            2500,
            3432
        ]
        if frame in frame_map
    ]


    test_frames = st.multiselect(
        "Select frames",
        options=frames,
        default=default_test_frames
    )


    if st.button(
        "Preview Saved ROI"
    ):

        if not test_frames:

            st.warning(
                "Select at least one frame."
            )

        else:

            for frame in test_frames:

                image_path = frame_map[
                    frame
                ]

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:

                    st.warning(
                        f"Could not read frame {frame}."
                    )

                    continue


                cropped = crop_with_roi(
                    image,
                    saved_roi
                )


                st.write(
                    f"### Frame {frame}"
                )

                st.image(
                    cropped,
                    use_container_width=True,
                    clamp=True
                )


# ============================================================
# FINAL STATUS
# ============================================================

st.divider()

if saved_roi is not None:

    st.success(
        "✓ ROI selection complete. "
        "The same ROI can now be passed automatically "
        "to the DIC engine."
    )

else:

    st.info(
        "Draw an ROI and click "
        "'Confirm and Save ROI'."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "The ROI is selected visually with the mouse. "
    "The saved coordinates are mapped to the original "
    "image dimensions so they can be reused by the DIC engine."
)