import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'Dataset'
STAGE9 = DATASET / 'Stage9_Full_Sequential_DIC'
CSV_PATH = STAGE9 / 'dic_1_2.csv'
DASHBOARD_PATH = Path(__file__).with_name('stage18_final_dashboard_v7.py')
ROI_PATH = DATASET / 'Stage20_ROI_Selection' / 'roi.json'
OUTPUT_PATH = ROOT / 'results' / 'diagnostics' / 'dic_1_2_displacement_magnitude.png'
REQUIRED_COLUMNS = ['X', 'Y', 'dx', 'dy', 'ZNCC']
MIN_ZNCC = 0.95
SUPPORT_RADIUS = 35


def dashboard_helpers():
    tree = ast.parse(DASHBOARD_PATH.read_text(encoding='utf-8'))
    wanted = {'local_strain', 'heatmap_figure'}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError('Could not find dashboard local_strain/heatmap_figure helpers.')

    namespace = {
        'np': np,
        'pd': pd,
        'cKDTree': cKDTree,
        'griddata': griddata,
        'gaussian_filter': gaussian_filter,
        'plt': plt,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(DASHBOARD_PATH), 'exec'), namespace)
    return namespace['local_strain'], namespace['heatmap_figure']


def frame_number(path):
    numbers = re.findall(r'\d+', path.name)
    return int(numbers[-2]) if len(numbers) >= 2 else int(numbers[-1])


def print_distribution(label, values):
    values = np.asarray(values, dtype=float)
    print(
        f'{label}: min={np.min(values):.10g}, max={np.max(values):.10g}, '
        f'mean={np.mean(values):.10g}, median={np.median(values):.10g}, '
        f'std={np.std(values, ddof=1):.10g}'
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'Missing existing Stage 9 result: {CSV_PATH}')
    if not ROI_PATH.exists():
        raise FileNotFoundError(f'Missing saved ROI metadata: {ROI_PATH}')

    raw = pd.read_csv(CSV_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError('Stage 9 CSV is missing columns: ' + ', '.join(missing))
    numeric = raw[REQUIRED_COLUMNS].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric.to_numpy()).all(axis=1)
    valid = numeric.loc[finite & (numeric['ZNCC'] >= MIN_ZNCC)].copy()
    if valid.empty:
        raise ValueError('No finite pair 1 -> 2 points passed the ZNCC threshold.')

    magnitude = np.hypot(valid['dx'].to_numpy(), valid['dy'].to_numpy())
    print('DIC pair: 1 -> 2')
    print(f'Number of DIC points: {len(raw):,}')
    print(f'Number of valid points (finite values, ZNCC >= {MIN_ZNCC:.2f}): {len(valid):,}')
    print_distribution('dx (pixels)', valid['dx'].to_numpy())
    print_distribution('dy (pixels)', valid['dy'].to_numpy())
    print(
        'Displacement magnitude (pixels): '
        f'min={np.min(magnitude):.10g}, max={np.max(magnitude):.10g}, '
        f'mean={np.mean(magnitude):.10g}, median={np.median(magnitude):.10g}'
    )
    print(
        f'ZNCC: mean={valid["ZNCC"].mean():.10g}, '
        f'min={valid["ZNCC"].min():.10g}'
    )

    roi = json.loads(ROI_PATH.read_text(encoding='utf-8'))
    reference_path = Path(roi['image_path'])
    image_folder = DATASET / 'BGSL'
    target_matches = [
        path for path in image_folder.iterdir()
        if path.suffix.lower() in {'.tif', '.tiff'} and frame_number(path) == 2
    ]
    if len(target_matches) != 1:
        raise ValueError(f'Expected one frame 2 TIFF, found {len(target_matches)}.')
    target_path = target_matches[0]
    x0, y0 = int(roi['x']), int(roi['y'])
    width, height = int(roi['width']), int(roi['height'])

    def read_roi(path):
        image = np.asarray(Image.open(path).convert('RGB'))
        return image[y0:y0 + height, x0:x0 + width]

    reference_roi = read_roi(reference_path)
    target_roi = read_roi(target_path)
    if reference_roi.shape[:2] != target_roi.shape[:2]:
        raise ValueError('Reference and target ROI dimensions do not match.')
    if np.max(valid['X']) >= width or np.max(valid['Y']) >= height:
        raise ValueError('DIC coordinates exceed the saved ROI dimensions.')

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    axes[0].imshow(reference_roi)
    axes[0].set_title('Reference frame 1')
    axes[1].imshow(target_roi)
    axes[1].set_title('Target frame 2')
    axes[2].imshow(target_roi)
    scatter = axes[2].scatter(
        valid['X'], valid['Y'], c=magnitude, cmap='magma',
        s=9, alpha=0.75, linewidths=0, vmin=0, vmax=0.25
    )
    axes[2].set_title('Target + DIC displacement magnitude')
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(height, 0)
    axes[2].set_xlabel('ROI X (pixels)')
    axes[2].set_ylabel('ROI Y (pixels)')
    fig.colorbar(scatter, ax=axes[2], label='Displacement magnitude (pixels)')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)
    print(f'Displacement-only visualization saved to: {OUTPUT_PATH}')

    local_strain, heatmap_figure = dashboard_helpers()
    strain = local_strain(valid)
    principal = strain['Principal_1'].to_numpy(float)
    print('\nREAL PAIR: local_affine principal strain')
    print_distribution('Displacement dx (pixels)', valid['dx'].to_numpy())
    print_distribution('Displacement dy (pixels)', valid['dy'].to_numpy())
    print(
        'Displacement magnitude (pixels): '
        f'mean={np.mean(magnitude):.10g}, median={np.median(magnitude):.10g}, '
        f'P95={np.percentile(magnitude, 95):.10g}, P99={np.percentile(magnitude, 99):.10g}, '
        f'max={np.max(magnitude):.10g}'
    )
    print(
        'Principal strain: '
        f'mean={np.mean(principal):.10g}, median={np.median(principal):.10g}, '
        f'P95={np.percentile(principal, 95):.10g}, '
        f'P99={np.percentile(principal, 99):.10g}, max={np.max(principal):.10g}'
    )

    zero = valid.copy()
    zero['dx'] = 0.0
    zero['dy'] = 0.0
    zero_strain = local_strain(zero)['Principal_1'].to_numpy(float)
    zero_stats = {
        'mean': float(np.mean(zero_strain)),
        'p95': float(np.percentile(zero_strain, 95)),
        'p99': float(np.percentile(zero_strain, 99)),
        'max': float(np.max(zero_strain)),
        'max_abs': float(np.max(np.abs(zero_strain))),
    }
    print('\nSynthetic zero-field:')
    print(
        f'mean strain={zero_stats["mean"]:.10g}, P95={zero_stats["p95"]:.10g}, '
        f'P99={zero_stats["p99"]:.10g}, maximum={zero_stats["max"]:.10g}'
    )
    if zero_stats['max_abs'] > 1e-10:
        print('FAIL: zero displacement produced non-zero strain; stopping before noise/interpolation tests.')
        raise SystemExit(2)
    print('PASS: zero displacement produces zero strain within 1e-10.')

    rng = np.random.default_rng(1202)
    noise_magnitude = rng.uniform(0.01, 0.05, len(valid))
    angle = rng.uniform(0, 2 * np.pi, len(valid))
    noise = valid.copy()
    noise['dx'] = noise_magnitude * np.cos(angle)
    noise['dy'] = noise_magnitude * np.sin(angle)
    noise_principal = local_strain(noise)['Principal_1'].to_numpy(float)
    print('\nSynthetic noise-field:')
    print(
        f'displacement magnitude mean={np.mean(noise_magnitude):.10g}, '
        f'min={np.min(noise_magnitude):.10g}, max={np.max(noise_magnitude):.10g}'
    )
    print(
        f'principal strain mean={np.mean(noise_principal):.10g}, '
        f'P95={np.percentile(noise_principal, 95):.10g}, '
        f'P99={np.percentile(noise_principal, 99):.10g}, '
        f'maximum={np.max(noise_principal):.10g}'
    )

    test_figure, raw_strain_values = heatmap_figure(
        target_roi, strain, support_radius=SUPPORT_RADIUS
    )
    axes_with_images = [axis for axis in test_figure.axes if axis.images]
    plotted = np.asarray(axes_with_images[0].images[1].get_array(), dtype=float)
    plt.close(test_figure)

    plot_x = strain['X'].to_numpy(float)
    plot_y = strain['Y'].to_numpy(float)
    plot_z = strain['Principal_1'].to_numpy(float)
    finite_strain = np.isfinite(plot_x) & np.isfinite(plot_y) & np.isfinite(plot_z)
    gx = np.arange(0, target_roi.shape[1], 5)
    gy = np.arange(0, target_roi.shape[0], 5)
    grid_x, grid_y = np.meshgrid(gx, gy)
    tree = cKDTree(np.column_stack([plot_x[finite_strain], plot_y[finite_strain]]))
    nearest_distance, _ = tree.query(
        np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1
    )
    support = nearest_distance.reshape(grid_x.shape) <= SUPPORT_RADIUS
    plotted_finite = np.isfinite(plotted)
    outside_support = int(np.count_nonzero(plotted_finite & ~support))
    plotted_values = plotted[plotted_finite]
    scale_low, scale_high = np.percentile(raw_strain_values, [5, 99])

    print('\nHeatmap interpolation/support inspection:')
    print('Interpolation method: linear griddata; nearest-neighbour extrapolation: not used.')
    print(
        f'Finite plotted grid cells: {plotted_finite.sum():,}/{plotted.size:,}; '
        f'finite cells outside {SUPPORT_RADIUS}px DIC support: {outside_support:,}'
    )
    print(
        'Plotted strain field: '
        f'min={np.min(plotted_values):.10g}, P95={np.percentile(plotted_values, 95):.10g}, '
        f'P99={np.percentile(plotted_values, 99):.10g}, max={np.max(plotted_values):.10g}'
    )
    print(
        f'Current display limits from raw strain P5/P99 (reported, not modified): '
        f'vmin={scale_low:.10g}, vmax={scale_high:.10g}'
    )
    if outside_support:
        print('FAIL: plotted values extend beyond the computed DIC spatial-support mask.')
    else:
        print('PASS: no finite plotted heatmap values lie outside the DIC spatial-support mask.')


if __name__ == '__main__':
    main()
