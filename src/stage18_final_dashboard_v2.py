
import os
import re
import glob
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# Optional lightweight click component.
# Install with:
#   python -m pip install streamlit-image-coordinates
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICK_COMPONENT_OK = True
except Exception:
    CLICK_COMPONENT_OK = False


st.set_page_config(
    page_title="DIC Concrete Beam Analyzer",
    page_icon="📐",
    layout="wide",
)

st.title("📐 DIC Concrete Beam Analysis System")
st.caption(
    "Automated image–experiment synchronization, DIC deformation analysis, "
    "local strain heatmap, graphs and deformation-event screening."
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def natural_frame_number(filename):
    """
    BGSL-2-0001_0.tif -> 1
    Uses the second-last numeric group when possible.
    """
    name = Path(filename).name
    nums = re.findall(r"\d+", name)
    if len(nums) >= 2:
        return int(nums[-2])
    if nums:
        return int(nums[-1])
    return None


def find_image_for_frame(folder, frame):
    patterns = [
        str(Path(folder) / f"BGSL-2-{frame:04d}_0.tif"),
        str(Path(folder) / f"BGSL-2-{frame:04d}_0.TIF"),
        str(Path(folder) / f"*{frame:04d}_0.tif"),
        str(Path(folder) / f"*{frame:04d}_0.TIF"),
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]

    # Fallback: inspect names once.
    for p in glob.glob(str(Path(folder) / "*.tif")) + glob.glob(str(Path(folder) / "*.TIF")):
        if natural_frame_number(p) == frame:
            return p
    return None


def build_frame_index(folder):
    files = glob.glob(str(Path(folder) / "*.tif")) + glob.glob(str(Path(folder) / "*.TIF"))
    result = {}
    for f in files:
        n = natural_frame_number(f)
        if n is not None and n != 0:
            result[n] = f
    return dict(sorted(result.items()))


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def read_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_roi(img, roi):
    x, y, w, h = [int(v) for v in roi]
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)
    return img[y:y2, x:x2]


def parse_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    required = [
        "Image Number",
        "Time_0",
        "Load ch 2 kN",
        "Displacement ch 7 µm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing Excel columns: " + ", ".join(missing))

    df = df.copy()
    for c in required:
        if c != "Image Number":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Image Number"] = pd.to_numeric(df["Image Number"], errors="coerce")
    df = df.dropna(subset=["Image Number"]).copy()
    df["Image Number"] = df["Image Number"].astype(int)
    df = df.sort_values("Image Number").drop_duplicates("Image Number")
    return df


def fast_dic(reference, target, subset=41, search=81, step=10, zncc_threshold=0.98):
    """
    Fast point-wise DIC using OpenCV normalized template matching.
    Returns X,Y,dx,dy,ZNCC.
    """
    if subset % 2 == 0:
        subset += 1
    if search < subset:
        search = subset + 10
    if search % 2 == 0:
        search += 1

    r = subset // 2
    sr = search // 2

    h, w = reference.shape
    rows = []

    for cy in range(sr, h - sr, step):
        for cx in range(sr, w - sr, step):
            template = reference[cy-r:cy+r+1, cx-r:cx+r+1]

            if template.size == 0 or np.var(template) < 500:
                continue

            search_img = target[cy-sr:cy+sr+1, cx-sr:cx+sr+1]
            if search_img.shape[0] < subset or search_img.shape[1] < subset:
                continue

            result = cv2.matchTemplate(
                search_img, template, cv2.TM_CCOEFF_NORMED
            )
            _, max_score, _, max_loc = cv2.minMaxLoc(result)

            if not np.isfinite(max_score) or max_score < zncc_threshold:
                continue

            best_x = max_loc[0]
            best_y = max_loc[1]

            # Integer displacement relative to search-window center.
            dx = float(best_x - (sr - r))
            dy = float(best_y - (sr - r))

            # Light subpixel refinement in x/y using neighboring scores.
            px, py = best_x, best_y

            if 0 < px < result.shape[1] - 1:
                a, b, c = result[0, px-1], result[0, px], result[0, px+1]
                denom = a - 2*b + c
                if abs(denom) > 1e-12:
                    dx += 0.5 * (a - c) / denom

            if 0 < py < result.shape[0] - 1:
                a = result[py-1, 0]
                b = result[py, 0]
                c = result[py+1, 0]
                denom = a - 2*b + c
                if abs(denom) > 1e-12:
                    dy += 0.5 * (a - c) / denom

            if abs(dx) > 5 or abs(dy) > 5:
                continue

            rows.append((cx, cy, dx, dy, float(max_score)))

    return pd.DataFrame(rows, columns=["X", "Y", "dx", "dy", "ZNCC"])


def calculate_strain(dic_df, grid_step=5, neighbors=25):
    """
    Local-affine strain estimation from DIC point displacement.
    Uses a weighted least-squares affine fit in each local neighborhood.
    """
    d = dic_df.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["ZNCC"] >= 0.95].copy()

    if len(d) < max(30, neighbors):
        raise ValueError("Not enough high-quality DIC points for strain calculation.")

    # Merge duplicate coordinates robustly.
    d = (
        d.groupby(["X", "Y"], as_index=False)
        .agg({"dx": "median", "dy": "median", "ZNCC": "median"})
    )

    pts = d[["X", "Y"]].to_numpy(float)
    u = d["dx"].to_numpy(float)
    v = d["dy"].to_numpy(float)

    tree = cKDTree(pts)
    out = []

    k = min(neighbors, len(d))

    for i, p in enumerate(pts):
        dist, idx = tree.query(p, k=k)
        if np.isscalar(dist):
            dist = np.array([dist])
            idx = np.array([idx])

        x = pts[idx, 0] - p[0]
        y = pts[idx, 1] - p[1]

        # Weight close points more strongly.
        ww = 1.0 / (dist + 2.0)
        A = np.column_stack([np.ones_like(x), x, y])

        try:
            W = np.diag(ww)
            au = np.linalg.lstsq(W @ A, W @ u[idx], rcond=None)[0]
            av = np.linalg.lstsq(W @ A, W @ v[idx], rcond=None)[0]

            du_dx, du_dy = au[1], au[2]
            dv_dx, dv_dy = av[1], av[2]

            exx = du_dx
            eyy = dv_dy
            exy = 0.5 * (du_dy + dv_dx)

            # Principal strains.
            avg = 0.5 * (exx + eyy)
            rad = math.sqrt(
                max(0.0, ((exx - eyy) * 0.5) ** 2 + exy ** 2)
            )
            p1 = avg + rad
            p2 = avg - rad

            out.append((p[0], p[1], exx, eyy, exy, p1, p2))
        except Exception:
            continue

    return pd.DataFrame(
        out,
        columns=["X", "Y", "exx", "eyy", "exy", "Principal_1", "Principal_2"],
    )


def make_heatmap(image_rgb, strain_df, alpha=0.72):
    """
    Create a localized DIC strain overlay.
    """
    h, w = image_rgb.shape[:2]

    if len(strain_df) < 20:
        raise ValueError("Too few strain points for heatmap.")

    z = strain_df["Principal_1"].to_numpy(float)
    x = strain_df["X"].to_numpy(float)
    y = strain_df["Y"].to_numpy(float)

    finite = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
    x, y, z = x[finite], y[finite], z[finite]

    if len(z) < 20:
        raise ValueError("Too few finite strain values.")

    # Robust limits avoid a single numerical outlier dominating the plot.
    vmin = float(np.percentile(z, 5))
    vmax = float(np.percentile(z, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    gx = np.arange(0, w, 5)
    gy = np.arange(0, h, 5)
    GX, GY = np.meshgrid(gx, gy)

    Z = griddata((x, y), z, (GX, GY), method="linear")

    # Fill outside convex hull with nearest interpolation.
    Zn = griddata((x, y), z, (GX, GY), method="nearest")
    Z = np.where(np.isfinite(Z), Z, Zn)

    Z = gaussian_filter(Z, sigma=1.2)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.imshow(image_rgb, cmap="gray")
    im = ax.imshow(
        Z,
        extent=[0, w, h, 0],
        cmap="turbo",
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("DIC Local-Affine Principal Strain Heatmap")
    plt.colorbar(im, ax=ax, label="Principal strain")
    plt.tight_layout()

    return fig, {
        "vmin": vmin,
        "vmax": vmax,
        "mean": float(np.mean(z)),
        "p95": float(np.percentile(z, 95)),
        "p99": float(np.percentile(z, 99)),
        "max": float(np.max(z)),
    }


def make_graphs(df):
    figs = []

    fig1, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["Time_0"], df["Load ch 2 kN"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (kN)")
    ax.set_title("Load vs Time")
    ax.grid(True, alpha=0.25)
    figs.append(("Load vs Time", fig1))

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["Time_0"], df["Displacement ch 7 µm"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement ch 7 (µm)")
    ax.set_title("Displacement vs Time")
    ax.grid(True, alpha=0.25)
    figs.append(("Displacement vs Time", fig2))

    fig3, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["Displacement ch 7 µm"], df["Load ch 2 kN"])
    ax.set_xlabel("Displacement ch 7 (µm)")
    ax.set_ylabel("Load (kN)")
    ax.set_title("Load vs Displacement")
    ax.grid(True, alpha=0.25)
    figs.append(("Load vs Displacement", fig3))

    return figs


def screen_events(df):
    """
    Simple transparent mechanical-event screening.
    This is screening, not crack confirmation.
    """
    d = df.sort_values("Image Number").copy()
    d["dLoad"] = d["Load ch 2 kN"].diff()
    d["dDisp"] = d["Displacement ch 7 µm"].diff()

    load_thr = d["dLoad"].abs().quantile(0.95)
    disp_thr = d["dDisp"].abs().quantile(0.95)

    ev = d[
        (d["dLoad"] < -load_thr) &
        (d["dDisp"] > disp_thr)
    ].copy()

    ev["Interpretation"] = "Mechanical transition candidate"
    return ev, float(load_thr), float(disp_thr)


def get_saved_stage17(base_dataset):
    candidates = [
        Path(base_dataset) / "Stage17_Final_Event_Synthesis" / "final_event_synthesis.csv",
        Path(base_dataset) / "Stage14_Strong_Event_Verification" / "verified_strong_events.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p), p
            except Exception:
                pass
    return None, None


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("1. Dataset")

default_root = r"C:\Users\shrama\Desktop\Dam updated dataset"

root_path = st.sidebar.text_input(
    "Project folder",
    value=default_root,
    help="Local Windows folder containing Dataset and experimental_data.",
)

dataset_folder = st.sidebar.text_input(
    "Image folder",
    value=str(Path(root_path) / "Dataset" / "BGSL"),
)

excel_path = st.sidebar.text_input(
    "Excel file path (optional)",
    value=str(
        Path(root_path)
        / "experimental_data"
        / "combined_image_load_data_0_to_6600.xlsx"
    ),
)

uploaded_excel = st.sidebar.file_uploader(
    "Or upload Excel",
    type=["xlsx", "xls"],
)

st.sidebar.header("2. DIC settings")
subset = st.sidebar.slider("Subset size", 21, 61, 41, step=2)
search = st.sidebar.slider("Search window", 41, 101, 81, step=2)
grid_step = st.sidebar.slider("Grid step", 5, 20, 10)
zncc_threshold = st.sidebar.slider("ZNCC threshold", 0.90, 0.995, 0.98, 0.005)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

if not Path(dataset_folder).exists():
    st.warning(
        "Image folder was not found. Enter the local image-folder path in the sidebar."
    )
    st.stop()

with st.spinner("Indexing images..."):
    frame_index = build_frame_index(dataset_folder)

if not frame_index:
    st.error("No TIFF images were found in the selected folder.")
    st.stop()

try:
    if uploaded_excel is not None:
        exp_df = parse_excel(uploaded_excel)
    elif Path(excel_path).exists():
        exp_df = parse_excel(excel_path)
    else:
        exp_df = None
except Exception as e:
    st.error(f"Excel loading error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------

st.subheader("Dataset overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Available images", f"{len(frame_index):,}")
c2.metric("First frame", str(min(frame_index)))
c3.metric("Last frame", str(max(frame_index)))
c4.metric("Excel rows", f"{len(exp_df):,}" if exp_df is not None else "Not loaded")

if exp_df is not None:
    common = sorted(set(frame_index).intersection(set(exp_df["Image Number"])))
    st.caption(
        f"Image/Excel synchronization: {len(common):,} common frame numbers. "
        "Frame 0 is excluded."
    )

# ---------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------

st.header("3. Select beam ROI")

available_frames = list(frame_index.keys())
default_ref = available_frames[min(0, len(available_frames)-1)]

ref_frame = st.selectbox(
    "Reference image",
    available_frames,
    index=available_frames.index(default_ref),
)

ref_path = frame_index[ref_frame]
ref_rgb_full = read_color(ref_path)

st.write(
    f"Reference: `{Path(ref_path).name}` — "
    f"{ref_rgb_full.shape[1]} × {ref_rgb_full.shape[0]} pixels"
)

if not CLICK_COMPONENT_OK:
    st.error(
        "ROI click component is not installed/compatible with this Streamlit environment."
    )
    st.code(
        "python -m pip install streamlit-image-coordinates",
        language="bash",
    )
    st.info(
        "After installing, restart the app. The ROI is selected by clicking two "
        "corners on the displayed image."
    )
    st.stop()

# Downscale only for display. Coordinates are mapped back to original image.
MAX_DISPLAY_WIDTH = 1100
scale = min(1.0, MAX_DISPLAY_WIDTH / ref_rgb_full.shape[1])
display_w = int(ref_rgb_full.shape[1] * scale)
display_h = int(ref_rgb_full.shape[0] * scale)

display_img = cv2.resize(
    ref_rgb_full,
    (display_w, display_h),
    interpolation=cv2.INTER_AREA,
)

st.write("Click the **top-left** corner of the beam ROI.")
click1 = streamlit_image_coordinates(display_img, key="roi_click_1")

if click1:
    st.success(f"First corner: ({click1['x']}, {click1['y']})")
    st.write("Now click the **bottom-right** corner of the beam ROI.")
    click2 = streamlit_image_coordinates(display_img, key="roi_click_2")
else:
    click2 = None

roi = None
if click1 and click2:
    x1 = int(min(click1["x"], click2["x"]) / scale)
    y1 = int(min(click1["y"], click2["y"]) / scale)
    x2 = int(max(click1["x"], click2["x"]) / scale)
    y2 = int(max(click1["y"], click2["y"]) / scale)

    x1 = max(0, min(x1, ref_rgb_full.shape[1] - 1))
    y1 = max(0, min(y1, ref_rgb_full.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, ref_rgb_full.shape[1]))
    y2 = max(y1 + 1, min(y2, ref_rgb_full.shape[0]))

    roi = (x1, y1, x2 - x1, y2 - y1)

    preview = ref_rgb_full.copy()
    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        max(2, ref_rgb_full.shape[1] // 1000),
    )

    st.image(preview, caption=f"Selected ROI: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
    st.session_state["selected_roi"] = roi
    st.success("ROI selected successfully.")

elif "selected_roi" in st.session_state:
    roi = st.session_state["selected_roi"]
    st.info(f"Using previous ROI: {roi}")

if roi is None:
    st.info("Select two ROI corners to continue.")
    st.stop()

# ---------------------------------------------------------------------
# DIC target
# ---------------------------------------------------------------------

st.header("4. DIC reference → target analysis")

remaining = [f for f in available_frames if f != ref_frame]

target_frame = st.selectbox(
    "Target image",
    remaining,
    index=0 if remaining else None,
)

target_path = frame_index[target_frame]

if st.button("🚀 Run DIC + strain heatmap", type="primary"):
    with st.spinner("Running fast DIC..."):
        t0 = time.time()

        ref_gray = crop_roi(read_gray(ref_path), roi)
        tar_gray = crop_roi(read_gray(target_path), roi)

        dic = fast_dic(
            ref_gray,
            tar_gray,
            subset=subset,
            search=search,
            step=grid_step,
            zncc_threshold=zncc_threshold,
        )

        elapsed = time.time() - t0

    if dic.empty:
        st.error(
            "No valid DIC points were found. Lower the ZNCC threshold slightly "
            "or choose a better textured ROI."
        )
        st.stop()

    st.session_state["dic_result"] = dic
    st.session_state["dic_ref_frame"] = ref_frame
    st.session_state["dic_target_frame"] = target_frame
    st.session_state["dic_roi"] = roi
    st.session_state["dic_elapsed"] = elapsed

dic = st.session_state.get("dic_result")

if dic is not None:
    st.success(
        f"DIC completed: {len(dic):,} points in "
        f"{st.session_state.get('dic_elapsed', 0):.2f} s"
    )

    # DIC coordinates are ROI-relative. Add ROI origin for full-image coordinates.
    rx, ry, rw, rh = st.session_state["dic_roi"]
    dic_display = dic.copy()
    dic_display["X_full"] = dic_display["X"] + rx
    dic_display["Y_full"] = dic_display["Y"] + ry
    dic_display["Magnitude"] = np.sqrt(
        dic_display["dx"]**2 + dic_display["dy"]**2
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Tracked points", f"{len(dic):,}")
    q2.metric("Mean ZNCC", f"{dic['ZNCC'].mean():.5f}")
    q3.metric("Mean displacement", f"{dic_display['Magnitude'].mean():.4f} px")
    q4.metric("Max displacement", f"{dic_display['Magnitude'].max():.4f} px")

    # Strain heatmap
    st.header("5. DIC strain heatmap")

    try:
        strain = calculate_strain(dic)
        strain_full = strain.copy()
        strain_full["X"] += rx
        strain_full["Y"] += ry

        # Heatmap should be plotted on the ROI image.
        ref_roi_rgb = ref_rgb_full[ry:ry+rh, rx:rx+rw]
        fig, stats = make_heatmap(ref_roi_rgb, strain)

        st.pyplot(fig, clear_figure=True)

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Mean principal strain", f"{stats['mean']:.6f}")
        h2.metric("P95", f"{stats['p95']:.6f}")
        h3.metric("P99", f"{stats['p99']:.6f}")
        h4.metric("Maximum", f"{stats['max']:.6f}")

        st.caption(
            "This visualization represents a DIC-derived local-affine strain field. "
            "A high-strain region is a deformation indicator; it is not by itself "
            "a confirmed crack segmentation."
        )

        # Download point results.
        csv_bytes = strain_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download strain CSV",
            csv_bytes,
            file_name=f"strain_{ref_frame}_{target_frame}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Heatmap generation failed: {e}")

# ---------------------------------------------------------------------
# Experimental graphs
# ---------------------------------------------------------------------

st.header("6. Experimental graphs")

if exp_df is None:
    st.warning("Load the experimental Excel file to generate synchronized graphs.")
else:
    graph_df = exp_df.copy()

    # If possible, show only frames represented by the selected experiment.
    graph_tabs = st.tabs(["Load vs Time", "Displacement vs Time", "Load vs Displacement"])

    figs = make_graphs(graph_df)

    for tab, (_, fig) in zip(graph_tabs, figs):
        with tab:
            st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------
# Event screening
# ---------------------------------------------------------------------

st.header("7. Deformation / mechanical event screening")

if exp_df is not None:
    events, load_thr, disp_thr = screen_events(exp_df)

    e1, e2 = st.columns(2)
    e1.metric("|ΔLoad| 95th-percentile threshold", f"{load_thr:.4f} kN")
    e2.metric("|ΔDisplacement| 95th-percentile threshold", f"{disp_thr:.4f} µm")

    if len(events):
        show_cols = [
            c for c in [
                "Image Number",
                "Time_0",
                "Load ch 2 kN",
                "Displacement ch 7 µm",
                "dLoad",
                "dDisp",
                "Interpretation",
            ] if c in events.columns
        ]
        st.dataframe(events[show_cols].head(100), use_container_width=True)
        st.caption(
            "These are mechanical transition candidates based on synchronized "
            "load/displacement changes. They are not automatically declared cracks."
        )
    else:
        st.info("No mechanical transition candidates crossed the current thresholds.")

# ---------------------------------------------------------------------
# Existing validated event synthesis
# ---------------------------------------------------------------------

st.header("8. Previously validated DIC events")

saved_events, saved_path = get_saved_stage17(
    str(Path(root_path) / "Dataset")
)

if saved_events is not None:
    st.caption(f"Loaded from: `{saved_path}`")
    preferred = [
        c for c in [
            "Frame_1", "Frame_2", "Time_1", "Time_2",
            "Load_Delta_kN", "Displacement_ch7_Delta_um",
            "DIC_Average_Magnitude", "DIC_Maximum_Magnitude",
            "Principal_Strain_P99", "Principal_Strain_Max",
            "Interpretation"
        ] if c in saved_events.columns
    ]
    if preferred:
        st.dataframe(
            saved_events[preferred],
            use_container_width=True,
        )
    else:
        st.dataframe(saved_events, use_container_width=True)
else:
    st.info(
        "No previous Stage 17 event-synthesis file was found in the selected "
        "project folder. The current app can still perform fresh DIC and "
        "experimental analysis."
    )

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()
st.caption(
    "Final-year project prototype • Fast OpenCV DIC • Local-affine strain "
    "visualization • Experimental Excel synchronization"
)