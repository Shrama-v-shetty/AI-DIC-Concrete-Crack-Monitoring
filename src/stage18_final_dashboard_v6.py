import re, glob, time
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

try:
    from streamlit_cropper import st_cropper
except Exception:
    st.error('ROI selector is required. Install it with: python -m pip install streamlit-cropper')
    st.stop()

st.set_page_config(page_title='DIC Concrete Beam Analysis', page_icon='📐', layout='wide')

# ----------------------------- helpers -----------------------------

def frame_number(path):
    nums = re.findall(r'\d+', Path(path).name)
    return int(nums[-2]) if len(nums) >= 2 else (int(nums[-1]) if nums else None)

@st.cache_data(show_spinner=False)
def index_images(folder):
    paths = glob.glob(str(Path(folder) / '*.tif')) + glob.glob(str(Path(folder) / '*.TIF'))
    result = {}
    for p in paths:
        n = frame_number(p)
        if n is not None and n != 0:
            result[n] = p
    return dict(sorted(result.items()))


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Unable to read image: {path}')
    return img


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Unable to read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_excel(source):
    df = pd.read_excel(source)
    required = ['Image Number', 'Time_0', 'Load ch 2 kN', 'Displacement ch 7 µm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing Excel columns: ' + ', '.join(missing))
    df = df.copy()
    df['Image Number'] = pd.to_numeric(df['Image Number'], errors='coerce')
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Image Number'])
    df['Image Number'] = df['Image Number'].astype(int)
    return df.sort_values('Image Number').drop_duplicates('Image Number').reset_index(drop=True)


def fast_dic(ref, target, subset=41, search=81, step=10, threshold=0.98):
    subset += (subset % 2 == 0)
    search += (search % 2 == 0)
    r = subset // 2
    sr = search // 2
    h, w = ref.shape
    rows = []
    for cy in range(sr, h - sr, step):
        for cx in range(sr, w - sr, step):
            template = ref[cy-r:cy+r+1, cx-r:cx+r+1]
            if template.size == 0 or np.var(template) < 500:
                continue
            search_img = target[cy-sr:cy+sr+1, cx-sr:cx+sr+1]
            corr = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(corr)
            if not np.isfinite(score) or score < threshold:
                continue
            dx = float(loc[0] - (sr-r))
            dy = float(loc[1] - (sr-r))
            if abs(dx) > 5 or abs(dy) > 5:
                continue
            rows.append((cx, cy, dx, dy, float(score)))
    return pd.DataFrame(rows, columns=['X', 'Y', 'dx', 'dy', 'ZNCC'])


def dic_summary(dic):
    if dic.empty:
        return {'Tracked_Points': 0, 'Average_ZNCC': np.nan, 'Average_dx': np.nan,
                'Average_dy': np.nan, 'Average_Magnitude': np.nan, 'Maximum_Magnitude': np.nan}
    mag = np.sqrt(dic.dx ** 2 + dic.dy ** 2)
    return {'Tracked_Points': len(dic), 'Average_ZNCC': float(dic.ZNCC.mean()),
            'Average_dx': float(dic.dx.mean()), 'Average_dy': float(dic.dy.mean()),
            'Average_Magnitude': float(mag.mean()), 'Maximum_Magnitude': float(mag.max())}


def local_strain(dic, neighbors=25):
    d = dic.replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[d.ZNCC >= 0.95]
    d = d.groupby(['X', 'Y'], as_index=False).agg({'dx': 'median', 'dy': 'median', 'ZNCC': 'median'})
    if len(d) < neighbors:
        raise ValueError('Not enough valid DIC points for local strain calculation.')
    pts = d[['X', 'Y']].to_numpy(float)
    u = d.dx.to_numpy(float)
    v = d.dy.to_numpy(float)
    tree = cKDTree(pts)
    rows = []
    for i, p0 in enumerate(pts):
        dist, idx = tree.query(p0, k=min(neighbors, len(d)))
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        x = pts[idx, 0] - p0[0]
        y = pts[idx, 1] - p0[1]
        A = np.column_stack([np.ones(len(x)), x, y])
        W = np.diag(1.0 / (dist + 2.0))
        try:
            au = np.linalg.lstsq(W @ A, W @ u[idx], rcond=None)[0]
            av = np.linalg.lstsq(W @ A, W @ v[idx], rcond=None)[0]
            exx, eyy = au[1], av[2]
            exy = 0.5 * (au[2] + av[1])
            mean = 0.5 * (exx + eyy)
            radius = np.sqrt(max(0.0, ((exx - eyy) * 0.5) ** 2 + exy ** 2))
            rows.append((p0[0], p0[1], exx, eyy, exy, mean + radius, mean - radius))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=['X', 'Y', 'exx', 'eyy', 'exy', 'Principal_1', 'Principal_2'])


def heatmap_figure(image, strain, support_radius=35):
    """Plot strain only where the DIC measurements provide spatial support."""
    h, w = image.shape[:2]
    x = strain.X.to_numpy(float)
    y = strain.Y.to_numpy(float)
    z = strain.Principal_1.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if len(z) < 10:
        raise ValueError('Not enough valid strain points for a heatmap.')

    gx = np.arange(0, w, 5)
    gy = np.arange(0, h, 5)
    GX, GY = np.meshgrid(gx, gy)

    # Linear interpolation only. Do not fill unsupported regions by nearest-neighbour extrapolation.
    Z = griddata((x, y), z, (GX, GY), method='linear')
    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    support = dist.reshape(GX.shape) <= support_radius
    Z[~support] = np.nan

    finite = np.isfinite(Z)
    if not np.any(finite):
        raise ValueError('No spatially supported strain field could be generated.')
    # Fill NaNs only for Gaussian filtering, then restore the support mask.
    fill_value = float(np.nanmedian(Z))
    Z_fill = np.where(np.isfinite(Z), Z, fill_value)
    Z_fill = gaussian_filter(Z_fill, 0.8)
    Z_fill[~support] = np.nan

    vals = z[np.isfinite(z)]
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 99)
    if hi <= lo:
        hi = lo + 1e-6

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.imshow(image, cmap='gray')
    im = ax.imshow(Z_fill, extent=[0, w, h, 0], cmap='turbo', alpha=0.72,
                   vmin=lo, vmax=hi)
    ax.scatter(x, y, s=2, c='white', alpha=0.10, linewidths=0)
    ax.set_title('DIC Local-Affine Principal Strain Heatmap')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    fig.colorbar(im, ax=ax, label='Principal strain')
    fig.tight_layout()
    return fig, vals


def cumulative_displacement_from_pairs(stage9_dir, ref_frame, target_frame, available_frames,
                                        min_zncc=0.98, min_coverage=0.5):
    """
    Build a cumulative deformation field by chaining the already-computed
    consecutive DIC results. This avoids the unreliable direct ref->target
    matching for large frame gaps.

    The existing pair CSVs are accumulated on their common DIC coordinate grid.
    Missing/low-quality points are ignored, and only points with sufficient
    temporal coverage are retained.
    """
    if target_frame <= ref_frame:
        raise ValueError('Target frame must be greater than reference frame.')

    seq = [f for f in available_frames if ref_frame <= f <= target_frame]
    if not seq or seq[0] != ref_frame or seq[-1] != target_frame:
        raise ValueError('Reference and target must belong to the same available consecutive run.')
    if any(b != a + 1 for a, b in zip(seq[:-1], seq[1:])):
        raise ValueError('Reference and target cross a missing-image gap. Choose a target within the same consecutive run.')

    sum_dx = {}
    sum_dy = {}
    count = {}
    zncc_sum = {}
    pair_count = 0

    for f1, f2 in zip(seq[:-1], seq[1:]):
        csv_path = Path(stage9_dir) / f'dic_{f1}_{f2}.csv'
        if not csv_path.exists():
            continue
        d = pd.read_csv(csv_path)
        required = {'X', 'Y', 'dx', 'dy', 'ZNCC'}
        if not required.issubset(d.columns):
            continue
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['X','Y','dx','dy','ZNCC'])
        d = d[d['ZNCC'] >= min_zncc].copy()
        if d.empty:
            continue
        d = d.groupby(['X','Y'], as_index=False).agg({'dx':'median','dy':'median','ZNCC':'median'})
        pair_count += 1
        for row in d.itertuples(index=False):
            key = (float(row.X), float(row.Y))
            sum_dx[key] = sum_dx.get(key, 0.0) + float(row.dx)
            sum_dy[key] = sum_dy.get(key, 0.0) + float(row.dy)
            count[key] = count.get(key, 0) + 1
            zncc_sum[key] = zncc_sum.get(key, 0.0) + float(row.ZNCC)

    if pair_count == 0 or not count:
        raise ValueError('No usable consecutive DIC CSVs were found for this interval.')

    needed = max(1, int(np.ceil(pair_count * min_coverage)))
    rows = []
    for (x, y), n in count.items():
        if n >= needed:
            rows.append((x, y, sum_dx[(x,y)], sum_dy[(x,y)], zncc_sum[(x,y)] / n, n))

    if len(rows) < 25:
        raise ValueError(f'Only {len(rows)} cumulative DIC points survived temporal coverage filtering.')

    return pd.DataFrame(rows, columns=['X','Y','dx','dy','ZNCC','Coverage_Pairs']), pair_count


@st.cache_data(show_spinner=False)
def run_reference_dic(ref_path, target_path, roi, subset, search, step, threshold):
    """Run direct reference-to-target DIC for cumulative deformation visualization."""
    rx, ry, rw, rh = roi
    ref = read_gray(ref_path)[ry:ry + rh, rx:rx + rw]
    target = read_gray(target_path)[ry:ry + rh, rx:rx + rw]
    return fast_dic(ref, target, subset, search, step, threshold)

# ----------------------------- paths -----------------------------

st.title('📐 DIC Concrete Beam Analysis System')
st.caption('Full image sequence DIC • manual beam ROI • Excel synchronization • deformation progression • spatial strain • event detection')

st.sidebar.header('Dataset')
project_folder = st.sidebar.text_input('Project folder', r'C:\Users\shrama\Desktop\Dam updated dataset')
image_folder = st.sidebar.text_input('Image folder', str(Path(project_folder) / 'Dataset' / 'BGSL'))
excel_path = st.sidebar.text_input('Excel file', str(Path(project_folder) / 'experimental_data' / 'combined_image_load_data_0_to_6600.xlsx'))
excel_upload = st.sidebar.file_uploader('Or upload Excel', type=['xlsx', 'xls'])

st.sidebar.header('Fast DIC settings')
subset = st.sidebar.slider('Subset size', 21, 61, 41, 2)
search = st.sidebar.slider('Search window', 41, 101, 81, 2)
step = st.sidebar.slider('Grid step', 5, 20, 10)
threshold = st.sidebar.slider('ZNCC threshold', 0.90, 0.995, 0.98, 0.005)

if not Path(image_folder).exists():
    st.warning('Image folder not found. Check the path in the sidebar.')
    st.stop()

frames = index_images(image_folder)
if not frames:
    st.error('No TIFF images were found in the selected folder.')
    st.stop()

try:
    experiment = load_excel(excel_upload if excel_upload is not None else excel_path)
except Exception as e:
    st.error(f'Excel loading error: {e}')
    st.stop()

available = list(frames.keys())
pairs = [(a, b) for a, b in zip(available[:-1], available[1:]) if b == a + 1]
dataset_root = Path(project_folder) / 'Dataset'
stage9 = dataset_root / 'Stage9_Full_Sequential_DIC'
stage9.mkdir(parents=True, exist_ok=True)
summary_path = stage9 / 'full_dic_summary.csv'

# ----------------------------- overview -----------------------------

st.header('Dataset Overview')
a, b, c, d = st.columns(4)
a.metric('Available images', f'{len(available):,}')
b.metric('First frame', str(min(available)))
c.metric('Last frame', str(max(available)))
d.metric('Excel rows', f'{len(experiment):,}')
st.caption(f'Frame 0 is excluded. The system will process {len(pairs):,} available consecutive pairs and automatically skip missing ranges.')

# ----------------------------- ROI -----------------------------

st.header('1. Select Beam ROI')
roi_reference = st.selectbox('Choose an image for ROI selection', available, index=0)
roi_image_path = frames[roi_reference]
roi_image = read_rgb(roi_image_path)
st.info('Drag a rectangle around the concrete beam surface. You do not need to enter X/Y/width/height manually.')

crop = st_cropper(Image.fromarray(roi_image), realtime_update=True, box_color='#ff0000', aspect_ratio=None, return_type='box')
if crop is None:
    st.stop()

rx, ry = int(crop['left']), int(crop['top'])
rw, rh = int(crop['width']), int(crop['height'])
if rw <= 1 or rh <= 1:
    st.warning('Please select a valid ROI.')
    st.stop()

st.session_state['roi'] = (rx, ry, rw, rh)
st.success(f'Selected beam ROI: x={rx}, y={ry}, width={rw}, height={rh}')

# ----------------------------- full DIC -----------------------------

st.header('2. Full Dataset DIC Analysis')
st.info('The system analyses every available consecutive pair: 1→2, 2→3, …, 2525→2526, 2708→2709, …, 6599→6600. It never tries to match distant images such as 1→6600.')

run = st.button('🚀 Process Full Available Image Sequence', type='primary')

if run:
    progress = st.progress(0.0)
    status = st.empty()
    records = []
    t_start = time.time()
    for i, (f1, f2) in enumerate(pairs):
        status.write(f'Processing {i + 1:,}/{len(pairs):,}: {f1} → {f2}')
        out_csv = stage9 / f'dic_{f1}_{f2}.csv'
        try:
            if out_csv.exists():
                dic = pd.read_csv(out_csv)
            else:
                ref = read_gray(frames[f1])
                target = read_gray(frames[f2])
                ref = ref[ry:ry + rh, rx:rx + rw]
                target = target[ry:ry + rh, rx:rx + rw]
                t0 = time.time()
                dic = fast_dic(ref, target, subset, search, step, threshold)
                dic.to_csv(out_csv, index=False)
            s = dic_summary(dic)
            s.update({'Frame_1': f1, 'Frame_2': f2})
            records.append(s)
        except Exception as e:
            records.append({'Frame_1': f1, 'Frame_2': f2, 'Tracked_Points': 0, 'Average_ZNCC': np.nan, 'Average_dx': np.nan, 'Average_dy': np.nan, 'Average_Magnitude': np.nan, 'Maximum_Magnitude': np.nan, 'Error': str(e)})
        progress.progress((i + 1) / len(pairs))

    summary = pd.DataFrame(records)
    e1 = experiment[['Image Number', 'Time_0', 'Load ch 2 kN', 'Displacement ch 7 µm']].rename(columns={'Image Number': 'Frame_1', 'Time_0': 'Time_1', 'Load ch 2 kN': 'Load_1_kN', 'Displacement ch 7 µm': 'Displacement_1_um'})
    e2 = experiment[['Image Number', 'Time_0', 'Load ch 2 kN', 'Displacement ch 7 µm']].rename(columns={'Image Number': 'Frame_2', 'Time_0': 'Time_2', 'Load ch 2 kN': 'Load_2_kN', 'Displacement ch 7 µm': 'Displacement_2_um'})
    summary = summary.merge(e1, on='Frame_1', how='left').merge(e2, on='Frame_2', how='left')
    summary['Load_Delta_kN'] = summary['Load_2_kN'] - summary['Load_1_kN']
    summary['Displacement_Delta_um'] = summary['Displacement_2_um'] - summary['Displacement_1_um']
    summary.to_csv(summary_path, index=False)
    st.session_state['full_summary'] = summary
    status.success(f'Completed {len(summary):,} consecutive pairs in {(time.time() - t_start) / 60:.1f} minutes.')

if 'full_summary' not in st.session_state and summary_path.exists():
    st.session_state['full_summary'] = pd.read_csv(summary_path)
summary = st.session_state.get('full_summary')

if summary is not None and len(summary):
    good = summary[summary['Tracked_Points'] > 0]
    st.subheader('Full-sequence DIC results')
    a, b, c, d = st.columns(4)
    a.metric('DIC pairs', f'{len(summary):,}')
    b.metric('Successful pairs', f'{len(good):,}')
    c.metric('Mean ZNCC', f'{good.Average_ZNCC.mean():.5f}' if len(good) else '—')
    d.metric('Mean DIC magnitude', f'{good.Average_Magnitude.mean():.4f} px' if len(good) else '—')
    st.dataframe(summary.head(100), use_container_width=True)
    st.download_button('⬇️ Download full DIC + Excel summary', summary.to_csv(index=False).encode(), 'full_dic_excel_integrated_summary.csv', 'text/csv')

# ----------------------------- graphs -----------------------------

st.header('3. Experimental and DIC Progression')
tabs = st.tabs(['Load vs Time', 'Displacement vs Time', 'Load vs Displacement', 'DIC Magnitude', 'DIC Quality'])
with tabs[0]:
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(experiment['Time_0'], experiment['Load ch 2 kN']); ax.set_xlabel('Time'); ax.set_ylabel('Load (kN)'); ax.set_title('Load vs Time'); ax.grid(alpha=.25); st.pyplot(fig, clear_figure=True)
with tabs[1]:
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(experiment['Time_0'], experiment['Displacement ch 7 µm']); ax.set_xlabel('Time'); ax.set_ylabel('Displacement ch 7 (µm)'); ax.set_title('Displacement vs Time'); ax.grid(alpha=.25); st.pyplot(fig, clear_figure=True)
with tabs[2]:
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(experiment['Displacement ch 7 µm'], experiment['Load ch 2 kN']); ax.set_xlabel('Displacement ch 7 (µm)'); ax.set_ylabel('Load (kN)'); ax.set_title('Load vs Displacement'); ax.grid(alpha=.25); st.pyplot(fig, clear_figure=True)
if summary is not None and len(summary):
    with tabs[3]:
        fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(summary['Frame_2'], summary['Average_Magnitude']); ax.set_xlabel('Frame'); ax.set_ylabel('Average DIC magnitude (pixels)'); ax.set_title('DIC Deformation Progression'); ax.grid(alpha=.25); st.pyplot(fig, clear_figure=True)
    with tabs[4]:
        fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(summary['Frame_2'], summary['Average_ZNCC']); ax.set_xlabel('Frame'); ax.set_ylabel('Average ZNCC'); ax.set_title('DIC Correlation Quality'); ax.grid(alpha=.25); st.pyplot(fig, clear_figure=True)

# ----------------------------- spatial heatmaps -----------------------------

st.header('4. Spatial DIC Strain Heatmaps')

heat_tabs = st.tabs(['Consecutive-pair heatmap', 'Reference → target cumulative heatmap'])

with heat_tabs[0]:
    pair_files = [p for p in stage9.glob('dic_*.csv') if re.fullmatch(r'dic_\d+_\d+\.csv', p.name)]
    pair_files.sort(key=lambda p: tuple(map(int, re.findall(r'\d+', p.name))))
    if pair_files:
        labels = []
        for p in pair_files:
            nums = re.findall(r'\d+', p.name)
            labels.append(f'{nums[-2]} → {nums[-1]}')
        chosen_label = st.selectbox('Select any processed consecutive pair', labels, key='pair_heatmap')
        chosen = pair_files[labels.index(chosen_label)]
        if st.button('Generate Consecutive-Pair Heatmap'):
            nums = list(map(int, re.findall(r'\d+', chosen.name)))
            f1, f2 = nums[-2], nums[-1]
            try:
                dic = pd.read_csv(chosen)
                strain = local_strain(dic)
                target = read_rgb(frames[f2])
                target_roi = target[ry:ry + rh, rx:rx + rw]
                fig, z = heatmap_figure(target_roi, strain)
                st.pyplot(fig, clear_figure=True)
                a, b, c, d = st.columns(4)
                a.metric('Mean strain', f'{np.mean(z):.6f}')
                b.metric('P95', f'{np.percentile(z, 95):.6f}')
                c.metric('P99', f'{np.percentile(z, 99):.6f}')
                d.metric('Maximum', f'{np.max(z):.6f}')
                strain['X_full'] = strain['X'] + rx
                strain['Y_full'] = strain['Y'] + ry
                st.download_button('⬇️ Download strain CSV', strain.to_csv(index=False).encode(), f'strain_{f1}_{f2}.csv', 'text/csv')
            except Exception as e:
                st.error(f'Heatmap generation failed: {e}')
    else:
        st.info('Run the full sequence first. Then a processed consecutive pair can be selected.')

with heat_tabs[1]:
    st.info(
        'Cumulative mode now chains the existing consecutive-frame DIC results. '
        'It does NOT perform a direct Frame 1 → distant-frame match. This preserves '
        'small, reliable DIC steps while accumulating deformation up to the selected target.'
    )

    # Identify contiguous runs. A cumulative reference/target pair must stay
    # inside one run because the dataset has missing image ranges.
    runs = []
    current = []
    for f in available:
        if not current or f == current[-1] + 1:
            current.append(f)
        else:
            if current:
                runs.append(current)
            current = [f]
    if current:
        runs.append(current)

    run_labels = [f'Frames {r[0]} → {r[-1]} ({len(r):,} images)' for r in runs]
    selected_run_label = st.selectbox('Available consecutive run', run_labels, key='cumulative_run')
    selected_run = runs[run_labels.index(selected_run_label)]

    ref_frame = st.selectbox(
        'Reference frame',
        selected_run[:-1],
        index=0,
        key='cumulative_ref_v6'
    )
    target_options = [f for f in selected_run if f > ref_frame]

    if target_options:
        target_frame = st.selectbox(
            'Target frame',
            target_options,
            index=len(target_options)-1 if len(target_options) > 1 else 0,
            key='cumulative_target_v6'
        )

        c1, c2 = st.columns(2)
        with c1:
            cum_threshold = st.slider(
                'Minimum pair ZNCC', 0.90, 0.995, 0.98, 0.005,
                key='cum_threshold_v6'
            )
        with c2:
            cum_coverage = st.slider(
                'Minimum temporal coverage', 0.25, 1.00, 0.50, 0.05,
                key='cum_coverage_v6'
            )

        st.caption(
            f'Accumulating {target_frame - ref_frame:,} consecutive DIC intervals: '
            f'{ref_frame}→{ref_frame+1}→…→{target_frame}. '
            'Existing Stage 9 CSVs are reused; no full DIC recalculation is required.'
        )

        if st.button('Generate Cumulative Heatmap from Existing DIC', type='primary', key='cum_button_v6'):
            try:
                t0 = time.time()
                cumulative_dic, used_pairs = cumulative_displacement_from_pairs(
                    stage9, ref_frame, target_frame, selected_run,
                    min_zncc=cum_threshold, min_coverage=cum_coverage
                )
                elapsed = time.time() - t0

                strain_cum = local_strain(cumulative_dic)

                target_rgb = read_rgb(frames[target_frame])
                target_roi = target_rgb[ry:ry + rh, rx:rx + rw]

                # Plot cumulative strain at the deformed/current coordinates so
                # the field remains spatially aligned with the target image.
                strain_plot = strain_cum.copy()
                lookup = cumulative_dic.set_index(['X','Y'])
                strain_plot['X_plot'] = strain_plot.apply(
                    lambda r: r.X + float(lookup.loc[(r.X,r.Y), 'dx']) if (r.X,r.Y) in lookup.index else r.X, axis=1
                )
                strain_plot['Y_plot'] = strain_plot.apply(
                    lambda r: r.Y + float(lookup.loc[(r.X,r.Y), 'dy']) if (r.X,r.Y) in lookup.index else r.Y, axis=1
                )

                # Heatmap renderer expects X/Y. Use the current/deformed locations.
                display_strain = strain_plot.rename(columns={'X_plot':'X','Y_plot':'Y'})
                fig, z = heatmap_figure(target_roi, display_strain, support_radius=40)
                st.pyplot(fig, clear_figure=True)

                a, b, c, d = st.columns(4)
                a.metric('Consecutive DIC intervals', f'{used_pairs:,}')
                b.metric('Cumulative DIC points', f'{len(cumulative_dic):,}')
                c.metric('P99 cumulative strain', f'{np.percentile(z, 99):.6f}')
                d.metric('Maximum cumulative strain', f'{np.max(z):.6f}')

                st.success(
                    f'Cumulative DIC built from {used_pairs:,} existing consecutive pairs in {elapsed:.1f} s. '
                    'No direct long-range image matching was used.'
                )
                st.caption(
                    'The heatmap represents accumulated DIC-derived deformation/strain. '
                    'It is not a crack segmentation; the visible crack remains in the target image.'
                )

                cumulative_dic['X_full'] = cumulative_dic['X'] + rx
                cumulative_dic['Y_full'] = cumulative_dic['Y'] + ry
                cumulative_dic['Cumulative_Magnitude'] = np.sqrt(
                    cumulative_dic['dx']**2 + cumulative_dic['dy']**2
                )
                st.download_button(
                    '⬇️ Download cumulative DIC field',
                    cumulative_dic.to_csv(index=False).encode(),
                    f'cumulative_dic_{ref_frame}_{target_frame}.csv',
                    'text/csv'
                )
            except Exception as e:
                st.error(f'Cumulative heatmap generation failed: {e}')
    else:
        st.info('Choose a reference frame with at least one later frame in the same consecutive run.')

# ----------------------------- event detection -----------------------------

st.header('5. Correlated Deformation Events')
if summary is not None and len(summary):
    d = summary.copy()
    d['Abs_Load_Change'] = d['Load_Delta_kN'].abs()
    d['Abs_Displacement_Change'] = d['Displacement_Delta_um'].abs()
    lt = d['Abs_Load_Change'].quantile(.95)
    dt = d['Abs_Displacement_Change'].quantile(.95)
    mt = d['Average_Magnitude'].quantile(.95)
    events = d[(d['Load_Delta_kN'] < -lt) & (d['Displacement_Delta_um'] > dt) & (d['Average_Magnitude'] > mt)]
    a, b, c = st.columns(3); a.metric('95th |ΔLoad|', f'{lt:.4f} kN'); b.metric('95th |ΔCh7|', f'{dt:.4f} µm'); c.metric('95th DIC magnitude', f'{mt:.4f} px')
    if len(events):
        cols = [c for c in ['Frame_1','Frame_2','Time_1','Time_2','Load_Delta_kN','Displacement_Delta_um','Average_Magnitude','Maximum_Magnitude','Average_ZNCC','Tracked_Points'] if c in events.columns]
        st.dataframe(events[cols], use_container_width=True)
    else:
        st.info('No intervals crossed all three current thresholds.')

# ----------------------------- validated project events -----------------------------

st.header('6. Previously Validated Project Events')
stage17 = dataset_root / 'Stage17_Final_Event_Synthesis' / 'final_event_synthesis.csv'
if stage17.exists():
    final_events = pd.read_csv(stage17)
    st.dataframe(final_events, use_container_width=True)
    st.caption('These are previously validated project-level events. A localized deformation event is not automatically a confirmed crack.')
else:
    st.info('Stage 17 event synthesis was not found.')

st.divider()
st.success('Final workflow: image dataset → manual ROI → every available consecutive image pair → DIC → Excel synchronization → deformation progression → selectable spatial heatmaps → correlated event detection.')
