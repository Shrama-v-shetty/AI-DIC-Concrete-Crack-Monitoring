import os
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# STAGE 22 - DIC + EXCEL INTEGRATION DASHBOARD
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"

STAGE21_DIR = os.path.join(
    BASE_DIR,
    "Stage21_ROI_DIC_Integration"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Stage22_DIC_Excel_Integration"
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
# HEADER
# ============================================================

st.title(
    "DIC Concrete Beam Monitoring System"
)

st.caption(
    "Stage 22 — DIC + experimental data integration"
)

st.divider()


# ============================================================
# FIND STAGE 21 SUMMARY
# ============================================================

SUMMARY_CSV = os.path.join(
    STAGE21_DIR,
    "roi_dic_summary.csv"
)


if not os.path.exists(
    SUMMARY_CSV
):

    st.error(
        "Stage 21 DIC summary was not found.\n\n"
        f"{SUMMARY_CSV}"
    )

    st.info(
        "Run Stage 21 first and complete at least "
        "one DIC test."
    )

    st.stop()


# ============================================================
# LOAD DIC SUMMARY
# ============================================================

@st.cache_data
def load_dic_summary(path):

    return pd.read_csv(
        path
    )


dic_df = load_dic_summary(
    SUMMARY_CSV
)


# ============================================================
# DIC SUMMARY
# ============================================================

st.subheader(
    "DIC Results"
)

if dic_df.empty:

    st.warning(
        "The Stage 21 DIC summary is empty."
    )

    st.stop()


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "DIC pairs",
    f"{len(dic_df):,}"
)

c2.metric(
    "Average tracked points",
    f"{int(dic_df['Tracked_Points'].mean()):,}"
)

c3.metric(
    "Average ZNCC",
    f"{dic_df['Average_ZNCC'].mean():.6f}"
)

c4.metric(
    "Average DIC magnitude",
    f"{dic_df['Average_Magnitude'].mean():.6f}"
)


with st.expander(
    "View DIC pair results"
):

    st.dataframe(
        dic_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXCEL INPUT
# ============================================================

st.divider()

st.subheader(
    "Experimental Excel"
)

uploaded_excel = st.file_uploader(
    "Upload the experimental Excel file",
    type=[
        "xlsx",
        "xls"
    ]
)


# ============================================================
# EXCEL VALIDATION
# ============================================================

if uploaded_excel is None:

    st.info(
        "Upload the experimental Excel file to "
        "synchronize it with the DIC results."
    )

    st.stop()


try:

    excel_df = pd.read_excel(
        uploaded_excel
    )

except Exception as error:

    st.error(
        f"Could not read the Excel file:\n{error}"
    )

    st.stop()


required_columns = [
    "Image Number",
    "Time_0",
    "Load ch 2 kN",
    "Displacement ch 7 µm"
]


missing_columns = [
    column
    for column in required_columns
    if column not in excel_df.columns
]


if missing_columns:

    st.error(
        "The uploaded Excel is missing:\n"
        + ", ".join(
            missing_columns
        )
    )

    st.stop()


# ============================================================
# CLEAN EXCEL
# ============================================================

excel_df[
    "Image Number"
] = pd.to_numeric(
    excel_df[
        "Image Number"
    ],
    errors="coerce"
)

excel_df[
    "Time_0"
] = pd.to_numeric(
    excel_df[
        "Time_0"
    ],
    errors="coerce"
)

excel_df[
    "Load ch 2 kN"
] = pd.to_numeric(
    excel_df[
        "Load ch 2 kN"
    ],
    errors="coerce"
)

excel_df[
    "Displacement ch 7 µm"
] = pd.to_numeric(
    excel_df[
        "Displacement ch 7 µm"
    ],
    errors="coerce"
)


excel_df = excel_df[
    excel_df[
        "Image Number"
    ].notna()
].copy()


excel_df[
    "Image Number"
] = excel_df[
    "Image Number"
].astype(int)


# ============================================================
# CREATE FRAME LOOKUP
# ============================================================

excel_lookup = (
    excel_df
    .drop_duplicates(
        subset=["Image Number"],
        keep="first"
    )
    .set_index(
        "Image Number"
    )
)


# ============================================================
# INTEGRATE DIC + EXCEL
# ============================================================

integrated_rows = []

for _, dic_row in dic_df.iterrows():

    frame1 = int(
        dic_row["Frame_1"]
    )

    frame2 = int(
        dic_row["Frame_2"]
    )


    if (
        frame1 not in excel_lookup.index
        or
        frame2 not in excel_lookup.index
    ):

        continue


    exp1 = excel_lookup.loc[
        frame1
    ]

    exp2 = excel_lookup.loc[
        frame2
    ]


    time1 = float(
        exp1["Time_0"]
    )

    time2 = float(
        exp2["Time_0"]
    )

    load1 = float(
        exp1["Load ch 2 kN"]
    )

    load2 = float(
        exp2["Load ch 2 kN"]
    )

    disp1 = float(
        exp1["Displacement ch 7 µm"]
    )

    disp2 = float(
        exp2["Displacement ch 7 µm"]
    )


    integrated_rows.append(
        {
            "Frame_1":
                frame1,

            "Frame_2":
                frame2,

            "Time_1":
                time1,

            "Time_2":
                time2,

            "Time_Delta_sec":
                time2 - time1,

            "Load_1_kN":
                load1,

            "Load_2_kN":
                load2,

            "Load_Delta_kN":
                load2 - load1,

            "Displacement_ch7_1_um":
                disp1,

            "Displacement_ch7_2_um":
                disp2,

            "Displacement_ch7_Delta_um":
                disp2 - disp1,

            "Tracked_Points":
                dic_row["Tracked_Points"],

            "DIC_Average_ZNCC":
                dic_row["Average_ZNCC"],

            "DIC_Average_dx":
                dic_row["Average_dx"],

            "DIC_Average_dy":
                dic_row["Average_dy"],

            "DIC_Average_Magnitude":
                dic_row["Average_Magnitude"],

            "DIC_Maximum_Magnitude":
                dic_row["Maximum_Magnitude"],

            "DIC_Processing_Time_sec":
                dic_row["Processing_Time_sec"]
        }
    )


integrated_df = pd.DataFrame(
    integrated_rows
)


# ============================================================
# MATCHING SUMMARY
# ============================================================

st.divider()

st.subheader(
    "Frame Synchronization"
)

matched_pairs = len(
    integrated_df
)

total_dic_pairs = len(
    dic_df
)


m1, m2, m3 = st.columns(3)

m1.metric(
    "DIC pairs",
    f"{total_dic_pairs:,}"
)

m2.metric(
    "Matched pairs",
    f"{matched_pairs:,}"
)

m3.metric(
    "Unmatched pairs",
    f"{total_dic_pairs - matched_pairs:,}"
)


if matched_pairs == 0:

    st.error(
        "No DIC pairs could be matched with the "
        "experimental Excel."
    )

    st.stop()


if matched_pairs == total_dic_pairs:

    st.success(
        "All DIC pairs were successfully synchronized "
        "with the experimental data."
    )

else:

    st.warning(
        "Some DIC pairs could not be synchronized."
    )


# ============================================================
# INTEGRATED DATASET
# ============================================================

st.divider()

st.subheader(
    "Integrated DIC + Experimental Dataset"
)

st.dataframe(
    integrated_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SAVE INTEGRATED DATASET
# ============================================================

integrated_csv = os.path.join(
    OUTPUT_DIR,
    "integrated_dic_experimental_data.csv"
)

integrated_df.to_csv(
    integrated_csv,
    index=False
)


st.success(
    "Integrated dataset created successfully."
)

st.code(
    integrated_csv
)


# ============================================================
# EXPERIMENTAL RESPONSE GRAPHS
# ============================================================

st.divider()

st.subheader(
    "Experimental Response Graphs"
)


tab1, tab2, tab3 = st.tabs(
    [
        "Load vs Time",
        "Displacement vs Time",
        "Load vs Displacement"
    ]
)


# ------------------------------------------------------------
# LOAD VS TIME
# ------------------------------------------------------------

with tab1:

    plot_df = excel_df.sort_values(
        "Time_0"
    )

    st.line_chart(
        plot_df.set_index(
            "Time_0"
        )[
            [
                "Load ch 2 kN"
            ]
        ]
    )


# ------------------------------------------------------------
# DISPLACEMENT VS TIME
# ------------------------------------------------------------

with tab2:

    plot_df = excel_df.sort_values(
        "Time_0"
    )

    st.line_chart(
        plot_df.set_index(
            "Time_0"
        )[
            [
                "Displacement ch 7 µm"
            ]
        ]
    )


# ------------------------------------------------------------
# LOAD VS DISPLACEMENT
# ------------------------------------------------------------

with tab3:

    chart_df = excel_df[
        [
            "Displacement ch 7 µm",
            "Load ch 2 kN"
        ]
    ].copy()

    chart_df = chart_df.dropna()

    st.scatter_chart(
        chart_df,
        x="Displacement ch 7 µm",
        y="Load ch 2 kN"
    )


# ============================================================
# DIC GRAPHS
# ============================================================

st.divider()

st.subheader(
    "DIC Response"
)


dic_tab1, dic_tab2 = st.tabs(
    [
        "DIC Deformation",
        "DIC Quality"
    ]
)


# ------------------------------------------------------------
# DIC DEFORMATION
# ------------------------------------------------------------

with dic_tab1:

    chart_df = integrated_df[
        [
            "Frame_2",
            "DIC_Average_Magnitude"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "Frame_2"
    )

    st.line_chart(
        chart_df
    )


# ------------------------------------------------------------
# DIC QUALITY
# ------------------------------------------------------------

with dic_tab2:

    chart_df = integrated_df[
        [
            "Frame_2",
            "DIC_Average_ZNCC"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "Frame_2"
    )

    st.line_chart(
        chart_df
    )


# ============================================================
# MECHANICAL TRANSITION IDENTIFICATION
# ============================================================

st.divider()

st.subheader(
    "Mechanical Transitions"
)


threshold_load = float(
    np.percentile(
        np.abs(
            integrated_df[
                "Load_Delta_kN"
            ]
        ),
        95
    )
)


threshold_disp = float(
    np.percentile(
        np.abs(
            integrated_df[
                "Displacement_ch7_Delta_um"
            ]
        ),
        95
    )
)


integrated_df[
    "Significant_Load_Change"
] = (
    np.abs(
        integrated_df[
            "Load_Delta_kN"
        ]
    )
    >= threshold_load
)


integrated_df[
    "Significant_Displacement_Change"
] = (
    np.abs(
        integrated_df[
            "Displacement_ch7_Delta_um"
        ]
    )
    >= threshold_disp
)


integrated_df[
    "Load_Drop_Displacement_Rise"
] = (
    integrated_df[
        "Load_Delta_kN"
    ] < 0
) & (
    integrated_df[
        "Displacement_ch7_Delta_um"
    ] > 0
)


integrated_df[
    "Mechanical_Transition"
] = (
    integrated_df[
        "Significant_Load_Change"
    ]
    &
    integrated_df[
        "Significant_Displacement_Change"
    ]
)


print_events = integrated_df[
    integrated_df[
        "Load_Drop_Displacement_Rise"
    ]
].copy()


q1, q2, q3 = st.columns(3)

q1.metric(
    "95% load-change threshold",
    f"{threshold_load:.4f} kN"
)

q2.metric(
    "95% Ch7-change threshold",
    f"{threshold_disp:.2f} µm"
)

q3.metric(
    "Load↓ + displacement↑",
    f"{len(print_events):,}"
)


if not print_events.empty:

    st.write(
        "Intervals with load decrease and "
        "displacement increase:"
    )

    st.dataframe(
        print_events[
            [
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
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No load-drop/displacement-rise intervals "
        "were present in the selected DIC range."
    )


# ============================================================
# SAVE UPDATED INTEGRATED DATA
# ============================================================

integrated_df.to_csv(
    integrated_csv,
    index=False
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.divider()

st.subheader(
    "Export"
)

with open(
    integrated_csv,
    "rb"
) as file:

    st.download_button(
        label="Download Integrated CSV",
        data=file,
        file_name="integrated_dic_experimental_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Experimental displacement channel 7 is used as "
    "the primary displacement measurement. DIC results "
    "are synchronized using image frame numbers. "
    "Load-drop/displacement-rise intervals indicate "
    "mechanical transitions and are not automatically "
    "labelled as cracks."
)