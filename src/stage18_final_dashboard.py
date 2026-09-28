import os
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# STAGE 18 - FINAL DIC MONITORING DASHBOARD
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"


# ------------------------------------------------------------
# DATA DIRECTORIES
# ------------------------------------------------------------

STAGE13_DIR = os.path.join(
    BASE_DIR,
    "Stage13_Final_Integrated_Analysis"
)

STAGE17_DIR = os.path.join(
    BASE_DIR,
    "Stage17_Final_Event_Synthesis"
)

STAGE12D_DIR = os.path.join(
    BASE_DIR,
    "Stage12D_Combined_Progression"
)

STAGE12A_DIR = os.path.join(
    BASE_DIR,
    "Stage12A_Significant_Region_Validation"
)

STAGE12B_DIR = os.path.join(
    BASE_DIR,
    "Stage12B_Significant_Region_Validation"
)

STAGE12C_DIR = os.path.join(
    BASE_DIR,
    "Stage12C_Significant_Region_Validation"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DIC Concrete Beam Monitoring System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Digital Image Correlation based deformation and "
    "experimental response analysis"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_stage13():

    path = os.path.join(
        STAGE13_DIR,
        "final_integrated_analysis.csv"
    )

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


@st.cache_data
def load_stage17():

    path = os.path.join(
        STAGE17_DIR,
        "final_event_synthesis.csv"
    )

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


@st.cache_data
def load_stage12():

    path = os.path.join(
        STAGE12D_DIR,
        "stage12_combined_progression.csv"
    )

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


stage13 = load_stage13()
stage17 = load_stage17()
stage12 = load_stage12()


# ============================================================
# CHECK DATA
# ============================================================

if stage13 is None:

    st.error(
        "Stage 13 integrated dataset was not found."
    )

    st.stop()


if stage17 is None:

    st.error(
        "Stage 17 event dataset was not found."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Analysis Controls"
)

st.sidebar.write(
    "Current dataset:"
)

st.sidebar.code(
    "Dam updated dataset"
)

st.sidebar.write(
    "Primary displacement:"
)

st.sidebar.code(
    "Displacement ch 7 (µm)"
)


# ============================================================
# TABS
# ============================================================

tab_summary, tab_graphs, tab_heatmaps, tab_events = (
    st.tabs(
        [
            "Summary",
            "Graphs",
            "DIC Heatmaps",
            "Events"
        ]
    )
)


# ============================================================
# SUMMARY TAB
# ============================================================

with tab_summary:

    st.subheader(
        "Experiment Overview"
    )

    total_intervals = len(stage13)

    load_drop_rise = int(
        stage13[
            "Load_Drop_Displacement_Rise"
        ].sum()
    )

    mechanical_events = int(
        stage13[
            "Correlated_Mechanical_Response"
        ].sum()
    )

    dic_candidates = int(
        (
            stage13[
                "Significant_DIC_Average"
            ]
            |
            stage13[
                "Significant_DIC_Maximum"
            ]
        ).sum()
    )

    strong_events = int(
        stage13[
            "Experimental_DIC_Correlation"
        ].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "DIC intervals",
        f"{total_intervals:,}"
    )

    col2.metric(
        "Load↓ + Displacement↑",
        f"{load_drop_rise:,}"
    )

    col3.metric(
        "DIC candidates",
        f"{dic_candidates:,}"
    )

    col4.metric(
        "Strong correlated events",
        f"{strong_events:,}"
    )


    st.divider()

    st.subheader(
        "Highest-Confidence Deformation Event"
    )

    localized = stage17[
        stage17[
            "Final_Interpretation"
        ]
        == "High-confidence localized deformation"
    ].copy()

    if len(localized) > 0:

        event = localized.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Frames",
            f"{int(event['Frame_1'])} → "
            f"{int(event['Frame_2'])}"
        )

        c2.metric(
            "Load change",
            f"{event['Load_Delta_kN']:.3f} kN"
        )

        c3.metric(
            "Ch7 displacement",
            f"+{event['Primary_Displacement_Delta_um']:.1f} µm"
        )

        c4.metric(
            "Localization score",
            f"{event['Localization_Score']:.3f}"
        )

        st.success(
            "High-confidence localized deformation detected. "
            "This is a deformation classification, not an "
            "automatic confirmed-crack label."
        )

        st.write(
            f"**High-strain centroid:** "
            f"({event['High_Strain_Centroid_X']:.1f}, "
            f"{event['High_Strain_Centroid_Y']:.1f})"
        )

        st.write(
            f"**DIC maximum deformation:** "
            f"{event['DIC_Maximum_Magnitude']:.4f} pixels"
        )

        st.write(
            f"**P99 principal strain:** "
            f"{event['Principal_Strain_P99']:.6f}"
        )

    else:

        st.info(
            "No high-confidence localized deformation "
            "event was identified."
        )


# ============================================================
# GRAPHS TAB
# ============================================================

with tab_graphs:

    st.subheader(
        "Experimental Response"
    )

    graph_options = [
        "Load vs Time",
        "Displacement vs Time",
        "Load vs Displacement",
        "DIC Deformation vs Frame",
        "DIC Quality vs Frame",
        "Validated DIC Strain Progression",
        "High-Strain Region Movement"
    ]

    selected_graph = st.selectbox(
        "Select graph",
        graph_options
    )


    graph_files = {

        "Load vs Time":
            os.path.join(
                STAGE13_DIR,
                "final_load_vs_time.png"
            ),

        "Displacement vs Time":
            os.path.join(
                STAGE13_DIR,
                "final_displacement_vs_time.png"
            ),

        "Load vs Displacement":
            os.path.join(
                STAGE13_DIR,
                "final_load_vs_displacement.png"
            ),

        "DIC Deformation vs Frame":
            os.path.join(
                STAGE13_DIR,
                "final_dic_deformation_vs_frame.png"
            ),

        "DIC Quality vs Frame":
            os.path.join(
                STAGE13_DIR,
                "final_dic_quality_vs_frame.png"
            ),

        "Validated DIC Strain Progression":
            os.path.join(
                STAGE13_DIR,
                "validated_strain_progression.png"
            ),

        "High-Strain Region Movement":
            os.path.join(
                STAGE13_DIR,
                "validated_centroid_progression.png"
            )
    }

    selected_file = graph_files[
        selected_graph
    ]

    if os.path.exists(
        selected_file
    ):

        st.image(
            Image.open(selected_file),
            use_container_width=True
        )

    else:

        st.warning(
            "Selected graph was not found."
        )


# ============================================================
# HEATMAP TAB
# ============================================================

with tab_heatmaps:

    st.subheader(
        "DIC Principal-Strain Heatmaps"
    )

    heatmap_options = {

        "Early: 1274 → 1500":
            os.path.join(
                STAGE12A_DIR,
                "heatmap_1274_1500.png"
            ),

        "Intermediate: 2000 → 2500":
            os.path.join(
                STAGE12B_DIR,
                "heatmap_2000_2500.png"
            ),

        "Advanced: 3000 → 3432":
            os.path.join(
                STAGE12C_DIR,
                "heatmap_3000_3432.png"
            )
    }

    selected_heatmap = st.selectbox(
        "Select deformation stage",
        list(heatmap_options.keys())
    )

    heatmap_file = heatmap_options[
        selected_heatmap
    ]

    if os.path.exists(
        heatmap_file
    ):

        st.image(
            Image.open(heatmap_file),
            use_container_width=True
        )

    else:

        st.warning(
            "Selected heatmap was not found."
        )


    st.divider()

    if stage12 is not None:

        st.subheader(
            "Validated Spatial Progression"
        )

        st.dataframe(
            stage12[
                [
                    "Frame_1",
                    "Frame_2",
                    "Mean_Principal_Strain",
                    "P99_Principal_Strain",
                    "Max_Principal_Strain",
                    "High_Strain_Centroid_X",
                    "High_Strain_Centroid_Y"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EVENTS TAB
# ============================================================

with tab_events:

    st.subheader(
        "Detected Deformation Events"
    )

    event_categories = [
        "All",
        "High-confidence localized deformation",
        "Central but spatially broad deformation",
        "Broad/global deformation"
    ]

    category = st.selectbox(
        "Filter events",
        event_categories
    )

    if category == "All":

        filtered = stage17.copy()

    else:

        filtered = stage17[
            stage17[
                "Final_Interpretation"
            ]
            == category
        ].copy()


    display_columns = [
        "Final_Event_Rank",
        "Frame_1",
        "Frame_2",
        "Time_1",
        "Time_2",
        "Load_Delta_kN",
        "Primary_Displacement_Delta_um",
        "DIC_Average_Magnitude",
        "DIC_Maximum_Magnitude",
        "DIC_Average_ZNCC",
        "Principal_Strain_P99",
        "Localization_Score",
        "Mechanical_Response",
        "DIC_Quality_Final",
        "Final_Interpretation"
    ]

    display_columns = [
        c
        for c in display_columns
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    st.subheader(
        "High-Confidence Localized Events"
    )

    high_conf = stage17[
        stage17[
            "Final_Interpretation"
        ]
        == "High-confidence localized deformation"
    ]

    if len(high_conf) == 0:

        st.info(
            "No high-confidence localized events."
        )

    else:

        st.dataframe(
            high_conf[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DIC deformation values are computed from image "
    "correlation. Experimental values are synchronized "
    "with the image/frame sequence. High-strain regions "
    "represent localized deformation and are not automatically "
    "classified as confirmed cracks."
)