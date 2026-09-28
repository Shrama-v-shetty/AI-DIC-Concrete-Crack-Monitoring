import re, glob, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
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


@st.cache_data(show_spinner=False)
def _cached_strain_grid(previous_frame, target_frame, dic_csv, csv_mtime_ns, csv_size, smoothing_sigma=2.0):
    if target_frame != previous_frame + 1:
        raise ValueError('The heatmap requires an immediately consecutive frame pair.')
    if Path(dic_csv).name != f'dic_{previous_frame}_{target_frame}.csv':
        raise ValueError('The selected Stage 9 CSV does not match the selected frame pair.')

    dic = pd.read_csv(dic_csv)
    required = ['X', 'Y', 'dx', 'dy', 'ZNCC']
    missing = [column for column in required if column not in dic.columns]
    if missing:
        raise ValueError('Stage 9 CSV is missing columns: ' + ', '.join(missing))
    numeric = dic[required].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric[['X', 'Y', 'dx', 'dy', 'ZNCC']].to_numpy()).all(axis=1)
    valid_points = numeric.loc[finite & (numeric['ZNCC'] >= 0.95)]
    if len(valid_points) < 3:
        raise ValueError('Not enough finite DIC points passed ZNCC >= 0.95.')

    points = valid_points[['X', 'Y']].to_numpy(float)
    dx_points = valid_points['dx'].to_numpy(float)
    dy_points = valid_points['dy'].to_numpy(float)
    grid_step = 10.0
    gx = np.arange(points[:, 0].min(), points[:, 0].max() + grid_step, grid_step)
    gy = np.arange(points[:, 1].min(), points[:, 1].max() + grid_step, grid_step)
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError('DIC point coverage is too small to calculate strain gradients.')
    grid_x, grid_y = np.meshgrid(gx, gy)

    dx_grid = griddata(points, dx_points, (grid_x, grid_y), method='linear')
    dy_grid = griddata(points, dy_points, (grid_x, grid_y), method='linear')

    # cKDTree is used only to exclude grid cells without nearby measured support.
    nearest_distance, _ = cKDTree(points).query(
        np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1
    )
    support = nearest_distance.reshape(grid_x.shape) <= 35.0
    valid_grid = support & np.isfinite(dx_grid) & np.isfinite(dy_grid)
    dx_grid[~valid_grid] = np.nan
    dy_grid[~valid_grid] = np.nan

    # Normalize by smoothed support weights, then restore unsupported cells as NaN.
    support_weights = gaussian_filter(valid_grid.astype(float), sigma=smoothing_sigma)
    smoothed_dx = gaussian_filter(np.where(valid_grid, dx_grid, 0.0), sigma=smoothing_sigma)
    smoothed_dy = gaussian_filter(np.where(valid_grid, dy_grid, 0.0), sigma=smoothing_sigma)
    dx_grid[valid_grid] = smoothed_dx[valid_grid] / support_weights[valid_grid]
    dy_grid[valid_grid] = smoothed_dy[valid_grid] / support_weights[valid_grid]
    dx_grid[~valid_grid] = np.nan
    dy_grid[~valid_grid] = np.nan

    du_dy, du_dx = np.gradient(dx_grid, gy, gx, edge_order=1)
    dv_dy, dv_dx = np.gradient(dy_grid, gy, gx, edge_order=1)

    f_xx = 1.0 + du_dx
    f_xy = du_dy
    f_yx = dv_dx
    f_yy = 1.0 + dv_dy
    e_xx = 0.5 * (f_xx * f_xx + f_yx * f_yx - 1.0)
    e_yy = 0.5 * (f_xy * f_xy + f_yy * f_yy - 1.0)
    e_xy = 0.5 * (f_xx * f_xy + f_yx * f_yy)
    mean_strain = 0.5 * (e_xx + e_yy)
    radius = np.sqrt((0.5 * (e_xx - e_yy)) ** 2 + e_xy ** 2)
    principal_max = mean_strain + radius
    principal_min = mean_strain - radius
    strain = np.where(
        valid_grid & np.isfinite(principal_max) & np.isfinite(principal_min),
        principal_max,
        np.nan,
    )

    values = strain[np.isfinite(strain)]
    if values.size == 0:
        raise ValueError('No finite supported principal-strain cells were produced.')
    maximum_index = np.nanargmax(strain)
    max_row, max_column = np.unravel_index(maximum_index, strain.shape)
    stats = {
        'mean': float(np.mean(values)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
        'maximum': float(strain[max_row, max_column]),
        'vmin': float(np.percentile(values, 5)),
        'vmax': float(np.percentile(values, 99)),
    }
    maximum_location = (float(gx[max_column]), float(gy[max_row]))
    return gx, gy, strain, stats, maximum_location


def calculate_strain_heatmap(previous_frame, target_frame, dic_csv, target_image, roi_origin=(0, 0)):
    csv_path = Path(dic_csv)
    stat = csv_path.stat()
    gx, gy, strain, strain_stats, strain_location = _cached_strain_grid(
        int(previous_frame), int(target_frame), str(csv_path), stat.st_mtime_ns,
        stat.st_size, smoothing_sigma=2.0,
    )

    grid_x, grid_y = np.meshgrid(gx, gy)
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    height, width = target_image.shape[:2]
    ax.imshow(target_image, extent=[0, width, height, 0], origin='upper')
    heatmap = ax.pcolormesh(
        grid_x, grid_y, np.ma.masked_invalid(strain), shading='nearest',
        cmap=cmap, vmin=strain_stats['vmin'],
        vmax=max(strain_stats['vmax'], strain_stats['vmin'] + 1e-12),
        alpha=0.72, rasterized=True,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f'Maximum Principal Strain | DIC pair: {previous_frame} → {target_frame}')
    ax.set_xlabel('ROI X (pixels)')
    ax.set_ylabel('ROI Y (pixels)')
    fig.colorbar(heatmap, ax=ax, label='Maximum Principal Strain')

    strain_location = (
        strain_location[0] + roi_origin[0],
        strain_location[1] + roi_origin[1],
    )
    strain_field = pd.DataFrame({
        'X': grid_x.ravel() + roi_origin[0],
        'Y': grid_y.ravel() + roi_origin[1],
        'Maximum_Principal_Strain': strain.ravel(),
    })
    return fig, strain_stats, strain_location, strain_field


def cumulative_displacement_from_pairs(stage9_dir, ref_frame, target_frame, available_frames,
                                        min_zncc=0.98, min_coverage=0.5):
    """
    Build a cleaned cumulative displacement field on the fixed Stage 9 grid.

    Each pair has global translation removed, then its residual displacement
    is lightly smoothed on the fixed DIC grid before accumulation.
    """
    if target_frame <= ref_frame:
        raise ValueError('Target frame must be greater than reference frame.')

    seq = [f for f in available_frames if ref_frame <= f <= target_frame]
    if not seq or seq[0] != ref_frame or seq[-1] != target_frame:
        raise ValueError('Reference and target must belong to the same available consecutive run.')
    if any(b != a + 1 for a, b in zip(seq[:-1], seq[1:])):
        raise ValueError('Reference and target cross a missing-image gap. Choose a target within the same consecutive run.')

    intervals = list(zip(seq[:-1], seq[1:]))
    sample_count = min(len(intervals), 5) if len(intervals) > 6 else len(intervals)
    interval_groups = np.array_split(np.arange(len(intervals)), sample_count)
    representative_pairs = [
        (*intervals[group[len(group) // 2]], len(group))
        for group in interval_groups
    ]

    pair_data = []
    grid_points = set()

    for f1, f2, represented_intervals in representative_pairs:
        csv_path = Path(stage9_dir) / f'dic_{f1}_{f2}.csv'
        if not csv_path.exists():
            continue

        d = pd.read_csv(csv_path)
        required = {'X', 'Y', 'dx', 'dy', 'ZNCC'}
        if not required.issubset(d.columns):
            continue

        d = d[list(required)].apply(pd.to_numeric, errors='coerce')
        d = d.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['X', 'Y', 'dx', 'dy', 'ZNCC']
        )
        d = d[d['ZNCC'] >= min_zncc].copy()
        if d.empty:
            continue

        # One displacement value per spatial point in this pair.
        d = d.groupby(['X', 'Y'], as_index=False).agg(
            {'dx': 'median', 'dy': 'median', 'ZNCC': 'median'}
        )

        d['dx'] -= float(d['dx'].median())
        d['dy'] -= float(d['dy'].median())
        d = d.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['X', 'Y', 'dx', 'dy', 'ZNCC']
        )
        if d.empty:
            continue
        pair_data.append((d, represented_intervals))
        grid_points.update(
            (float(row.X), float(row.Y)) for row in d.itertuples(index=False)
        )

    if not pair_data or not grid_points:
        raise ValueError('No usable consecutive DIC CSVs were found for this interval.')

    grid_points = sorted(grid_points)
    grid_x = np.unique([p[0] for p in grid_points])
    grid_y = np.unique([p[1] for p in grid_points])
    GX, GY = np.meshgrid(grid_x, grid_y)
    grid_index = {(float(x), float(y)): (iy, ix)
                  for iy, y in enumerate(grid_y)
                  for ix, x in enumerate(grid_x)}

    def smooth_increment(d, column):
        values = griddata(
            (d['X'].to_numpy(float), d['Y'].to_numpy(float)),
            d[column].to_numpy(float),
            (GX, GY),
            method='linear'
        )
        finite = np.isfinite(values)
        if not np.any(finite):
            return {}
        filled = np.where(finite, values, 0.0)
        weights = gaussian_filter(finite.astype(float), 1.25)
        smoothed = gaussian_filter(filled, 1.25) / np.maximum(weights, 1e-12)
        smoothed[~finite] = np.nan
        return {
            key: float(smoothed[iy, ix])
            for key, (iy, ix) in grid_index.items()
            if np.isfinite(smoothed[iy, ix])
        }

    sums = {
        key: {'dx': 0.0, 'dy': 0.0, 'coverage': 0, 'zncc_sum': 0.0}
        for key in grid_points
    }
    pair_count = 0
    for d, represented_intervals in pair_data:
        smoothed_dx = smooth_increment(d, 'dx')
        smoothed_dy = smooth_increment(d, 'dy')
        pair_count += 1
        quality = {(float(row.X), float(row.Y)): float(row.ZNCC)
                   for row in d.itertuples(index=False)}
        valid_keys = set(smoothed_dx) & set(smoothed_dy) & set(quality)
        for key in valid_keys:
            sums[key]['dx'] += smoothed_dx[key] * represented_intervals
            sums[key]['dy'] += smoothed_dy[key] * represented_intervals
            sums[key]['coverage'] += 1
            sums[key]['zncc_sum'] += quality.get(key, min_zncc)

    needed = max(1, int(np.ceil(len(representative_pairs) * min_coverage)))

    covered = [key for key, value in sums.items() if value['coverage'] >= needed]
    if len(covered) < 25:
        raise ValueError(
            f'Only {len(covered)} cumulative DIC points survived fixed-grid coverage '
            f'and coverage filtering.'
        )

    # Smooth the accumulated displacement once more, without adding points
    # outside the temporally supported fixed grid.
    covered_data = pd.DataFrame([
        (x, y, sums[(x, y)]['dx'], sums[(x, y)]['dy'])
        for x, y in covered
    ], columns=['X', 'Y', 'dx', 'dy'])
    final_dx = smooth_increment(covered_data, 'dx')
    final_dy = smooth_increment(covered_data, 'dy')

    rows = []
    for (x, y), s in sums.items():
        if s['coverage'] >= needed and (x, y) in final_dx and (x, y) in final_dy:
            rows.append((
                x,
                y,
                x,
                y,
                final_dx[(x, y)],
                final_dy[(x, y)],
                s['zncc_sum'] / s['coverage'],
                s['coverage'],
            ))

    return pd.DataFrame(
        rows,
        columns=[
            'X0', 'Y0', 'X', 'Y',
            'dx', 'dy', 'ZNCC', 'Coverage_Pairs'
        ]
    ), pair_count


def cumulative_heatmap_figure(image, cumulative_dic, support_radius=40):
    """
    Calculate and plot cumulative local-affine principal strain on the fixed
    original DIC grid. No nearest-neighbour extrapolation.
    """
    if cumulative_dic.empty:
        raise ValueError('Cumulative DIC field is empty.')

    # local_strain uses the original fixed-grid X/Y positions and cumulative
    # displacement values directly.
    strain = local_strain(
        cumulative_dic[['X', 'Y', 'dx', 'dy', 'ZNCC']]
    ).copy()

    # Plot only within the spatial support of the actual strain measurements.
    h, w = image.shape[:2]
    x = strain['X'].to_numpy(float)
    y = strain['Y'].to_numpy(float)
    z = strain['Principal_1'].to_numpy(float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    if len(z) < 10:
        raise ValueError('Not enough valid cumulative strain points for a heatmap.')

    gx = np.arange(0, w, 5)
    gy = np.arange(0, h, 5)
    GX, GY = np.meshgrid(gx, gy)

    Z = griddata((x, y), z, (GX, GY), method='linear')

    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(
        np.column_stack([GX.ravel(), GY.ravel()]), k=1
    )
    support = dist.reshape(GX.shape) <= support_radius
    Z[~support] = np.nan

    finite = np.isfinite(Z)
    if not np.any(finite):
        raise ValueError('No spatially supported cumulative strain field could be generated.')

    # Remove isolated single-cell strain spikes before light visualization smoothing.
    strain_fill = np.where(finite, Z, float(np.nanmedian(Z)))
    Z_filtered = median_filter(strain_fill, size=3, mode='nearest')
    Z_filtered[~finite | ~support] = np.nan
    finite_filtered = np.isfinite(Z_filtered)
    values = np.where(finite_filtered, Z_filtered, 0.0)
    weights = gaussian_filter(finite_filtered.astype(float), 0.8)
    Z_fill = gaussian_filter(values, 0.8) / np.maximum(weights, 1e-12)
    Z_fill[~finite_filtered | ~support] = np.nan

    vals = Z_filtered[np.isfinite(Z_filtered)]
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 99)
    if hi <= lo:
        hi = lo + 1e-6

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.imshow(image, cmap='gray')
    im = ax.imshow(
        Z_fill,
        extent=[0, w, h, 0],
        cmap='turbo',
        alpha=0.72,
        vmin=lo,
        vmax=hi
    )

    ax.scatter(x, y, s=2, c='white', alpha=0.10, linewidths=0)
    ax.set_title('Cumulative DIC Local-Affine Principal Strain Heatmap')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    fig.colorbar(im, ax=ax, label='Principal strain')
    fig.tight_layout()

    cumulative_x = cumulative_dic['X'].to_numpy(float)
    cumulative_y = cumulative_dic['Y'].to_numpy(float)
    cumulative_magnitude = np.sqrt(
        cumulative_dic['dx'].to_numpy(float) ** 2
        + cumulative_dic['dy'].to_numpy(float) ** 2
    )
    magnitude_valid = (
        np.isfinite(cumulative_x)
        & np.isfinite(cumulative_y)
        & np.isfinite(cumulative_magnitude)
    )
    M = griddata(
        (cumulative_x[magnitude_valid], cumulative_y[magnitude_valid]),
        cumulative_magnitude[magnitude_valid],
        (GX, GY),
        method='linear'
    )
    M[~support] = np.nan
    magnitude_fig, magnitude_ax = plt.subplots(figsize=(13, 5.5))
    magnitude_ax.imshow(image, cmap='gray')
    magnitude_image = magnitude_ax.imshow(
        M, extent=[0, w, h, 0], cmap='viridis', alpha=0.72
    )
    magnitude_ax.set_title('Debug: Cumulative Displacement Magnitude')
    magnitude_ax.set_xlabel('X (pixels)')
    magnitude_ax.set_ylabel('Y (pixels)')
    magnitude_fig.colorbar(magnitude_image, ax=magnitude_ax, label='Displacement magnitude (pixels)')
    magnitude_fig.tight_layout()

    strain_fig, strain_ax = plt.subplots(figsize=(13, 5.5))
    strain_image = strain_ax.imshow(
        Z_fill, extent=[0, w, h, 0], cmap='turbo', vmin=lo, vmax=hi
    )
    strain_ax.set_title('Debug: Cumulative Principal Strain')
    strain_ax.set_xlabel('X (pixels)')
    strain_ax.set_ylabel('Y (pixels)')
    strain_fig.colorbar(strain_image, ax=strain_ax, label='Principal strain')
    strain_fig.tight_layout()

    return fig, vals, strain, magnitude_fig, strain_fig


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

st.header('4. Spatial DIC Heatmaps')

heat_tabs = st.tabs(['Consecutive-pair heatmap', 'Reference → target cumulative heatmap'])

with heat_tabs[0]:
    available_pair_targets = [
        target for previous, target in pairs
        if (stage9 / f'dic_{previous}_{target}.csv').is_file()
    ]
    if available_pair_targets:
        target_frame = st.selectbox(
            'Select target frame for consecutive-pair heatmap',
            available_pair_targets,
            key='pair_heatmap_target',
        )
        previous_frame = target_frame - 1
        dic_csv = stage9 / f'dic_{previous_frame}_{target_frame}.csv'
        st.caption(f'DIC pair: {previous_frame} → {target_frame}')
        if st.button('Generate Consecutive-Pair Heatmap'):
            try:
                target = read_rgb(frames[target_frame])
                target_roi = target[ry:ry + rh, rx:rx + rw]
                fig, strain_stats, maximum_location, field_data = calculate_strain_heatmap(
                    previous_frame,
                    target_frame,
                    dic_csv,
                    target_roi,
                    roi_origin=(rx, ry),
                )
                st.subheader('Maximum Principal Strain')
                st.caption('DIC-derived maximum principal Green-Lagrange strain for the selected consecutive pair.')
                st.pyplot(fig, clear_figure=True)
                a, b, c, d, e, f = st.columns(6)
                a.metric('Mean strain', f'{strain_stats["mean"]:.6f}')
                b.metric('P95 strain', f'{strain_stats["p95"]:.6f}')
                c.metric('P99 strain', f'{strain_stats["p99"]:.6f}')
                d.metric('Maximum strain', f'{strain_stats["maximum"]:.6f}')
                e.metric('Maximum strain X', f'{maximum_location[0]:.1f} px')
                f.metric('Maximum strain Y', f'{maximum_location[1]:.1f} px')
                st.download_button(
                    '⬇️ Download strain CSV',
                    field_data.to_csv(index=False).encode(),
                    f'strain_{previous_frame}_{target_frame}.csv',
                    'text/csv',
                )
            except Exception as e:
                st.error(f'Heatmap generation failed: {e}')
    else:
        st.info('No available consecutive image pair has a matching Stage 9 DIC CSV.')

with heat_tabs[1]:
    st.info(
        'Cumulative mode samples and weights up to 5 existing adjacent-frame DIC results '
        'on the fixed DIC grid. This is a representative estimate, not a direct '
        'reference-to-distant-frame match or an exact sum of every frame pair.'
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
            f'The visualization samples up to 5 representative existing adjacent-pair CSVs '
            f'across {target_frame - ref_frame:,} frame intervals. Each sample is weighted '
            'by the span it represents; spatial support stays on the measured DIC grid. '
            'This is a representative estimate, not an exact sum of every frame pair.'
        )

        show_cumulative_debug = st.checkbox(
            'Show cumulative DIC debug fields', key='cum_debug_v6'
        )

        if st.button('Generate Cumulative Heatmap from Existing DIC', type='primary', key='cum_button_v6'):
            try:
                st.info('Generating cumulative heatmap using representative DIC intervals...')
                t0 = time.time()
                cumulative_dic, used_pairs = cumulative_displacement_from_pairs(
                    stage9, ref_frame, target_frame, selected_run,
                    min_zncc=cum_threshold, min_coverage=cum_coverage
                )
                elapsed = time.time() - t0

                target_rgb = read_rgb(frames[target_frame])
                target_roi = target_rgb[ry:ry + rh, rx:rx + rw]

                # Build strain directly from the fixed original DIC grid.
                # The cumulative field contains the summed displacement at
                # each original X/Y location.
                fig, z, strain_cum, magnitude_debug_fig, strain_debug_fig = cumulative_heatmap_figure(
                    target_roi,
                    cumulative_dic,
                    support_radius=40
                )

                if show_cumulative_debug:
                    st.subheader('Debug: Target Image with Cumulative Strain Overlay')
                st.pyplot(fig, clear_figure=True)
                if show_cumulative_debug:
                    st.subheader('Debug: Cumulative Displacement Magnitude')
                    st.pyplot(magnitude_debug_fig, clear_figure=True)
                    st.subheader('Debug: Cumulative Principal Strain')
                    st.pyplot(strain_debug_fig, clear_figure=True)


                a, b, c, d = st.columns(4)
                a.metric('Representative DIC samples', f'{used_pairs:,}')
                b.metric('Cumulative DIC points', f'{len(cumulative_dic):,}')
                c.metric('P99 cumulative strain', f'{np.percentile(z, 99):.6f}')
                d.metric('Maximum cumulative strain', f'{np.max(z):.6f}')

                st.success(
                    f'Cumulative DIC estimated from {used_pairs:,} representative existing pairs '
                    f'in {elapsed:.1f} s. No direct long-range image matching was used.'
                )
                st.caption(
                    'The heatmap represents accumulated DIC-derived deformation/strain. '
                    'It is not a crack segmentation; the visible crack remains in the target image.'
                )

                cumulative_dic['X_full'] = cumulative_dic['X'] + rx
                cumulative_dic['Y_full'] = cumulative_dic['Y'] + ry
                cumulative_dic['X0_full'] = cumulative_dic['X0'] + rx
                cumulative_dic['Y0_full'] = cumulative_dic['Y0'] + ry
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
