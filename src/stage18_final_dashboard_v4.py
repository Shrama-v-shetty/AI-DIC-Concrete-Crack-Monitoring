import os, re, glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except Exception:
    CROPPER_OK = False

st.set_page_config(
    page_title="DIC Concrete Beam Analysis",
    page_icon="📐",
    layout="wide"
)

# ============================================================
# PAGE
# ============================================================

st.title("📐 DIC Concrete Beam Analysis System")
st.caption(
    "Concrete beam deformation monitoring using DIC, experimental data "
    "synchronization and validated spatial strain stages."
)

# ============================================================
# HELPERS
# ============================================================

def get_frame_number(filename):
    nums = re.findall(r"\d+", Path(filename).name)
    if len(nums) >= 2:
        return int(nums[-2])
    return int(nums[-1]) if nums else None


@st.cache_data(show_spinner=False)
def index_images(folder):
    files = glob.glob(str(Path(folder) / "*.tif"))
    files += glob.glob(str(Path(folder) / "*.TIF"))

    result = {}
    for f in files:
        n = get_frame_number(f)
        if n is not None and n != 0:
            result[n] = f

    return dict(sorted(result.items()))


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image:\n{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_excel(source):
    df = pd.read_excel(source)

    required = [
        "Image Number",
        "Time_0",
        "Load ch 2 kN",
        "Displacement ch 7 µm"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The Excel file is missing:\n" + ", ".join(missing)
        )

    df = df.copy()
    df["Image Number"] = pd.to_numeric(
        df["Image Number"], errors="coerce"
    )

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Image Number"])
    df["Image Number"] = df["Image Number"].astype(int)

    return (
        df.sort_values("Image Number")
        .drop_duplicates("Image Number")
        .reset_index(drop=True)
    )


def display_heatmap(path, title):
    if not path.exists():
        st.warning(f"Validated heatmap not found:\n{path}")
        return False

    img = Image.open(path)
    st.image(
        img,
        caption=title,
        use_container_width=True
    )
    return True


def show_experimental_graphs(df):
    tabs = st.tabs([
        "Load vs Time",
        "Displacement vs Time",
        "Load vs Displacement"
    ])

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            df["Time_0"],
            df["Load ch 2 kN"]
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (kN)")
        ax.set_title("Load vs Time")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            df["Time_0"],
            df["Displacement ch 7 µm"]
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Displacement ch 7 (µm)")
        ax.set_title("Displacement vs Time")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    with tabs[2]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            df["Displacement ch 7 µm"],
            df["Load ch 2 kN"]
        )
        ax.set_xlabel("Displacement ch 7 (µm)")
        ax.set_ylabel("Load (kN)")
        ax.set_title("Load vs Displacement")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)


def event_screen(df):
    d = df.sort_values("Image Number").copy()

    d["ΔLoad (kN)"] = d["Load ch 2 kN"].diff()
    d["ΔDisplacement (µm)"] = d["Displacement ch 7 µm"].diff()

    load_threshold = float(
        d["ΔLoad (kN)"].abs().quantile(0.95)
    )
    displacement_threshold = float(
        d["ΔDisplacement (µm)"].abs().quantile(0.95)
    )

    events = d[
        (d["ΔLoad (kN)"] < -load_threshold) &
        (d["ΔDisplacement (µm)"] > displacement_threshold)
    ].copy()

    return events, load_threshold, displacement_threshold


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dataset")

project_folder = st.sidebar.text_input(
    "Project folder",
    value=r"C:\Users\shrama\Desktop\Dam updated dataset"
)

image_folder = st.sidebar.text_input(
    "Image folder",
    value=str(
        Path(project_folder) / "Dataset" / "BGSL"
    )
)

excel_path = st.sidebar.text_input(
    "Excel file",
    value=str(
        Path(project_folder)
        / "experimental_data"
        / "combined_image_load_data_0_to_6600.xlsx"
    )
)

uploaded_excel = st.sidebar.file_uploader(
    "Or upload Excel",
    type=["xlsx", "xls"]
)

# ============================================================
# LOAD DATA
# ============================================================

if not Path(image_folder).exists():
    st.warning(
        "Image folder not found. Please select the BGSL image folder."
    )
    st.stop()

frames = index_images(image_folder)

if not frames:
    st.error("No TIFF images were found in the selected folder.")
    st.stop()

try:
    if uploaded_excel is not None:
        experiment = load_excel(uploaded_excel)
    elif Path(excel_path).exists():
        experiment = load_excel(excel_path)
    else:
        experiment = None
except Exception as e:
    st.error(f"Excel loading error: {e}")
    st.stop()

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Available images",
    f"{len(frames):,}"
)

c2.metric(
    "First frame",
    str(min(frames))
)

c3.metric(
    "Last frame",
    str(max(frames))
)

c4.metric(
    "Excel rows",
    f"{len(experiment):,}" if experiment is not None else "Not loaded"
)

if experiment is not None:
    common_frames = set(frames).intersection(
        set(experiment["Image Number"])
    )

    st.caption(
        f"{len(common_frames):,} image numbers are synchronized "
        "with the experimental Excel data. Frame 0 is excluded."
    )

# ============================================================
# MANUAL ROI
# ============================================================

st.header("1. Select Beam ROI")

if not CROPPER_OK:
    st.error(
        "The manual ROI component is not installed."
    )
    st.code(
        "python -m pip install streamlit-cropper",
        language="bash"
    )
    st.stop()

frame_list = list(frames)

roi_reference = st.selectbox(
    "Select image for ROI",
    frame_list,
    index=0
)

roi_image_path = frames[roi_reference]
roi_image = read_rgb(roi_image_path)

st.write(
    f"Image: `{Path(roi_image_path).name}`  "
    f"({roi_image.shape[1]} × {roi_image.shape[0]} pixels)"
)

st.info(
    "Drag the rectangle around the concrete beam region. "
    "This ROI will be used as the analysis region."
)

crop_box = st_cropper(
    Image.fromarray(roi_image),
    realtime_update=True,
    box_color="#ff0000",
    aspect_ratio=None,
    return_type="box"
)

if crop_box is None:
    st.stop()

# streamlit-cropper returns a dictionary
roi_x = int(crop_box["left"])
roi_y = int(crop_box["top"])
roi_w = int(crop_box["width"])
roi_h = int(crop_box["height"])

if roi_w <= 1 or roi_h <= 1:
    st.warning("Please select a larger ROI.")
    st.stop()

st.session_state["roi"] = (
    roi_x,
    roi_y,
    roi_w,
    roi_h
)

preview = roi_image.copy()

cv2.rectangle(
    preview,
    (roi_x, roi_y),
    (roi_x + roi_w, roi_y + roi_h),
    (255, 0, 0),
    4
)

st.image(
    preview,
    caption=(
        f"Selected beam ROI — "
        f"x={roi_x}, y={roi_y}, "
        f"width={roi_w}, height={roi_h}"
    ),
    use_container_width=True
)

st.success("ROI selected successfully.")

# ============================================================
# VALIDATED DIC HEATMAP STAGES
# ============================================================

st.header("2. DIC Deformation Stages")

st.info(
    "For spatial strain visualization, the software uses the project's "
    "validated DIC progression stages. It does not directly match very "
    "distant images such as frame 1 → frame 6600, because large direct "
    "frame gaps can create false correspondences."
)

stage_options = {
    "Early deformation — 1274 → 1500":
        "Stage12A",
    "Middle deformation — 2000 → 2500":
        "Stage12B",
    "Late deformation — 3000 → 3432":
        "Stage12C",
}

selected_stage = st.selectbox(
    "Select deformation stage",
    list(stage_options.keys())
)

stage_folder = Path(project_folder) / "Dataset" / stage_options[selected_stage]

# Existing validated heatmap naming from the project.
possible_heatmaps = [
    stage_folder / "heatmap.png",
    stage_folder / f"heatmap_{selected_stage.split('—')[-1].strip().replace(' ', '')}.png",
]

# Explicit known project outputs.
if stage_options[selected_stage] == "Stage12A":
    possible_heatmaps += [
        Path(project_folder) / "Dataset" / "Stage12A" / "heatmap_1274_1500.png",
        Path(project_folder) / "Dataset" / "Stage12A" / "heatmap_1274_1500_strain.png",
    ]

elif stage_options[selected_stage] == "Stage12B":
    possible_heatmaps += [
        Path(project_folder) / "Dataset" / "Stage12B" / "heatmap_2000_2500.png",
        Path(project_folder) / "Dataset" / "Stage12B" / "heatmap_2000_2500_strain.png",
    ]

elif stage_options[selected_stage] == "Stage12C":
    possible_heatmaps += [
        Path(project_folder) / "Dataset" / "Stage12C" / "heatmap_3000_3432.png",
        Path(project_folder) / "Dataset" / "Stage12C" / "heatmap_3000_3432_strain.png",
    ]

heatmap_path = next(
    (p for p in possible_heatmaps if p.exists()),
    None
)

if heatmap_path is not None:
    display_heatmap(
        heatmap_path,
        selected_stage
    )
else:
    st.warning(
        "The validated heatmap image for this stage was not found."
    )

st.caption(
    "Heatmap stages are spatial DIC deformation visualizations. "
    "A high-strain region indicates localized deformation; it is not "
    "automatically classified as a confirmed crack."
)

# ============================================================
# EXPERIMENTAL GRAPHS
# ============================================================

st.header("3. Experimental Graphs")

if experiment is None:
    st.warning(
        "Load the Excel file to display the experimental graphs."
    )
else:
    show_experimental_graphs(experiment)

# ============================================================
# EVENT SCREENING
# ============================================================

st.header("4. Mechanical Event Screening")

if experiment is not None:
    events, load_thr, disp_thr = event_screen(experiment)

    e1, e2 = st.columns(2)

    e1.metric(
        "95th |ΔLoad| threshold",
        f"{load_thr:.4f} kN"
    )

    e2.metric(
        "95th |ΔDisplacement| threshold",
        f"{disp_thr:.4f} µm"
    )

    if len(events) > 0:
        columns = [
            "Image Number",
            "Time_0",
            "Load ch 2 kN",
            "Displacement ch 7 µm",
            "ΔLoad (kN)",
            "ΔDisplacement (µm)"
        ]

        st.dataframe(
            events[columns].head(100),
            use_container_width=True
        )

        st.caption(
            "These are mechanical transition candidates from the "
            "experimental load/displacement signals."
        )
    else:
        st.info(
            "No mechanical transition candidates crossed the "
            "current statistical thresholds."
        )

# ============================================================
# EXISTING FINAL EVENT SYNTHESIS
# ============================================================

st.header("5. Validated DIC Event Results")

stage17 = (
    Path(project_folder)
    / "Dataset"
    / "Stage17_Final_Event_Synthesis"
    / "final_event_synthesis.csv"
)

if stage17.exists():
    try:
        final_events = pd.read_csv(stage17)

        preferred_columns = [
            "Frame_1",
            "Frame_2",
            "Time_1",
            "Time_2",
            "Load_Delta_kN",
            "Displacement_ch7_Delta_um",
            "DIC_Average_Magnitude",
            "DIC_Maximum_Magnitude",
            "Principal_Strain_P99",
            "Principal_Strain_Max",
            "Centroid_X",
            "Centroid_Y",
            "Width",
            "Height",
            "Localization_Score",
            "Interpretation"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in final_events.columns
        ]

        if available_columns:
            st.dataframe(
                final_events[available_columns],
                use_container_width=True
            )
        else:
            st.dataframe(
                final_events,
                use_container_width=True
            )

        st.caption(
            "These are the previously validated project-level DIC event "
            "results. Localized deformation is not automatically a "
            "confirmed crack."
        )

    except Exception as e:
        st.warning(
            f"Could not load Stage 17 results: {e}"
        )
else:
    st.info(
        "Stage 17 final event synthesis was not found."
    )

# ============================================================
# DOWNLOAD / SUMMARY
# ============================================================

st.divider()

st.success(
    "Analysis workflow ready: dataset → Excel → manual ROI → "
    "validated DIC stage → experimental graphs → event results."
)

st.caption(
    "Final-year project prototype • Fast DIC • Manual beam ROI • "
    "Validated spatial deformation stages • Experimental synchronization"
)
