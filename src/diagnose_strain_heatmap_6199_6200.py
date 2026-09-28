from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'Dataset' / 'Stage9_Full_Sequential_DIC' / 'dic_6199_6200.csv'
ROI_IMAGE_PATH = PROJECT_ROOT / 'Dataset' / 'BGSL_ROI' / 'BGSL-2-6199_0.tif'
TARGET_IMAGE_PATH = PROJECT_ROOT / 'Dataset' / 'BGSL_ROI' / 'BGSL-2-6200_0.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'diagnostics'
RAW_OUTPUT_PATH = OUTPUT_DIR / 'strain_6199_6200_raw.png'
SMOOTHED_OUTPUT_PATH = OUTPUT_DIR / 'strain_6199_6200_smoothed.png'
OVERLAY_OUTPUT_PATH = OUTPUT_DIR / 'strain_6199_6200_overlay.png'

MIN_ZNCC = 0.95
GRID_STEP = 10.0
SUPPORT_RADIUS = 35.0
SMOOTH_SIGMA = 1.0


def smooth_supported_field(field):
	valid = np.isfinite(field)
	weights = gaussian_filter(valid.astype(float), sigma=SMOOTH_SIGMA)
	values = gaussian_filter(np.where(valid, field, 0.0), sigma=SMOOTH_SIGMA)
	smoothed = np.full(field.shape, np.nan, dtype=float)
	smoothed[valid] = values[valid] / weights[valid]
	return smoothed


def principal_strain(dx, dy, grid_x, grid_y):
	valid = np.isfinite(dx) & np.isfinite(dy)
	du_dy, du_dx = np.gradient(dx, grid_y[:, 0], grid_x[0, :], edge_order=1)
	dv_dy, dv_dx = np.gradient(dy, grid_y[:, 0], grid_x[0, :], edge_order=1)

	f_xx = 1.0 + du_dx
	f_xy = du_dy
	f_yx = dv_dx
	f_yy = 1.0 + dv_dy

	e_xx = 0.5 * (f_xx * f_xx + f_yx * f_yx - 1.0)
	e_yy = 0.5 * (f_xy * f_xy + f_yy * f_yy - 1.0)
	e_xy = 0.5 * (f_xx * f_xy + f_yx * f_yy)

	mean = 0.5 * (e_xx + e_yy)
	radius = np.sqrt((0.5 * (e_xx - e_yy)) ** 2 + e_xy ** 2)
	principal_max = mean + radius
	principal_min = mean - radius
	principal_max[~valid] = np.nan
	principal_min[~valid] = np.nan
	return principal_max, principal_min


def print_statistics(label, strain, grid_x, grid_y):
	valid = np.isfinite(strain)
	values = strain[valid]
	if values.size == 0:
		raise ValueError(f'{label} strain field contains no supported values.')

	maximum_index = np.nanargmax(strain)
	max_row, max_column = np.unravel_index(maximum_index, strain.shape)
	p99 = float(np.percentile(values, 99))
	p99_region = valid & (strain >= p99)
	centroid_x = float(np.mean(grid_x[p99_region]))
	centroid_y = float(np.mean(grid_y[p99_region]))
	stats = {
		'mean': float(np.mean(values)),
		'median': float(np.median(values)),
		'p95': float(np.percentile(values, 95)),
		'p99': p99,
		'maximum': float(values.max()),
	}
	maximum_location = (float(grid_x[max_row, max_column]), float(grid_y[max_row, max_column]))
	print(
		f'{label} principal strain: mean={stats["mean"]:.10g}, '
		f'median={stats["median"]:.10g}, P95={stats["p95"]:.10g}, '
		f'P99={stats["p99"]:.10g}, maximum={stats["maximum"]:.10g}'
	)
	print(f'{label} maximum coordinates: X={maximum_location[0]:.3f}, Y={maximum_location[1]:.3f}')
	print(f'{label} P99 region centroid: X={centroid_x:.3f}, Y={centroid_y:.3f}')
	return values, stats, maximum_location, (centroid_x, centroid_y)


def save_field_figure(field, grid_x, grid_y, vmin, vmax, path, title, image=None):
	masked = np.ma.masked_invalid(field)
	cmap = plt.get_cmap('turbo').copy()
	cmap.set_bad((0, 0, 0, 0))
	fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
	if image is not None:
		height, width = image.shape[:2]
		ax.imshow(image, extent=[0, width, height, 0], origin='upper', cmap='gray')
	mesh = ax.pcolormesh(
		grid_x, grid_y, masked, shading='nearest', cmap=cmap,
		vmin=vmin, vmax=vmax, alpha=0.76 if image is not None else 1.0,
		rasterized=True,
	)
	if image is not None:
		ax.set_xlim(0, image.shape[1])
		ax.set_ylim(image.shape[0], 0)
	else:
		ax.set_ylim(grid_y.max(), grid_y.min())
	ax.set_title(title)
	ax.set_xlabel('ROI X (pixels)')
	ax.set_ylabel('ROI Y (pixels)')
	fig.colorbar(mesh, ax=ax, label='Maximum principal Green-Lagrange strain')
	fig.savefig(path, dpi=180)
	plt.close(fig)


def main():
	for path, description in (
		(CSV_PATH, 'Stage 9 CSV'),
		(ROI_IMAGE_PATH, 'Reference image'),
		(TARGET_IMAGE_PATH, 'Target image'),
	):
		if not path.is_file():
			raise FileNotFoundError(f'{description} not found: {path}')

	raw = pd.read_csv(CSV_PATH)
	required = {'X', 'Y', 'dx', 'dy', 'ZNCC'}
	missing = sorted(required.difference(raw.columns))
	if missing:
		raise ValueError('DIC CSV is missing required columns: ' + ', '.join(missing))

	data = raw[list(required)].apply(pd.to_numeric, errors='coerce')
	finite = np.isfinite(data[['X', 'Y', 'dx', 'dy', 'ZNCC']].to_numpy()).all(axis=1)
	data = data.loc[finite & (data['ZNCC'] >= MIN_ZNCC)].copy()
	if data.empty:
		raise ValueError(f'No finite DIC points passed ZNCC >= {MIN_ZNCC:.2f}.')

	points = data[['X', 'Y']].to_numpy(float)
	dx_points = data['dx'].to_numpy(float)
	dy_points = data['dy'].to_numpy(float)
	grid_x_values = np.arange(data['X'].min(), data['X'].max() + GRID_STEP, GRID_STEP)
	grid_y_values = np.arange(data['Y'].min(), data['Y'].max() + GRID_STEP, GRID_STEP)
	grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
	dx_grid = griddata(points, dx_points, (grid_x, grid_y), method='linear')
	dy_grid = griddata(points, dy_points, (grid_x, grid_y), method='linear')

	# The tree is used only to determine whether interpolated cells have nearby measurements.
	nearest_distance, _ = cKDTree(points).query(
		np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1
	)
	supported = nearest_distance.reshape(grid_x.shape) <= SUPPORT_RADIUS
	valid = supported & np.isfinite(dx_grid) & np.isfinite(dy_grid)
	dx_grid[~valid] = np.nan
	dy_grid[~valid] = np.nan

	raw_max, _ = principal_strain(dx_grid, dy_grid, grid_x, grid_y)
	dx_smoothed = smooth_supported_field(dx_grid)
	dy_smoothed = smooth_supported_field(dy_grid)
	smoothed_max, _ = principal_strain(dx_smoothed, dy_smoothed, grid_x, grid_y)

	zncc_mean = float(data['ZNCC'].mean())
	zncc_min = float(data['ZNCC'].min())
	print(f'DIC pair: 6199 → 6200')
	print(f'Valid points: {len(data):,} of {len(raw):,}')
	print(f'ZNCC mean/min: {zncc_mean:.10g} / {zncc_min:.10g}')
	raw_values, raw_stats, raw_max_location, raw_p99_centroid = print_statistics(
		'Raw', raw_max, grid_x, grid_y
	)
	smoothed_values, smoothed_stats, maximum_location, p99_centroid = print_statistics(
		'Smoothed', smoothed_max, grid_x, grid_y
	)

	vmin = float(np.percentile(smoothed_values, 5))
	vmax = float(np.percentile(smoothed_values, 99))
	if vmax <= vmin:
		vmax = float(np.nextafter(vmin, np.inf))

	image = np.asarray(Image.open(ROI_IMAGE_PATH).convert('RGB'))
	target_image = np.asarray(Image.open(TARGET_IMAGE_PATH).convert('RGB'))
	if target_image.shape[:2] != image.shape[:2]:
		raise ValueError('Reference and target ROI image dimensions do not match.')
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	save_field_figure(
		raw_max, grid_x, grid_y, vmin, vmax, RAW_OUTPUT_PATH,
		'Raw maximum principal strain, pair 6199 -> 6200',
	)
	save_field_figure(
		smoothed_max, grid_x, grid_y, vmin, vmax, SMOOTHED_OUTPUT_PATH,
		'Smoothed maximum principal strain, pair 6199 -> 6200',
	)
	save_field_figure(
		smoothed_max, grid_x, grid_y, vmin, vmax, OVERLAY_OUTPUT_PATH,
		'Frame 6199 ROI with smoothed maximum principal strain', image=image,
	)
	print('\nSummary')
	print('DIC pair: 6199 → 6200')
	print(f'Valid points: {len(data):,}')
	print(f'ZNCC mean/min: {zncc_mean:.10g} / {zncc_min:.10g}')
	print('Raw principal strain:')
	print(f'  mean = {raw_stats["mean"]:.10g}')
	print(f'  median = {raw_stats["median"]:.10g}')
	print(f'  P95 = {raw_stats["p95"]:.10g}')
	print(f'  P99 = {raw_stats["p99"]:.10g}')
	print(f'  max = {raw_stats["maximum"]:.10g}')
	print('Smoothed principal strain:')
	print(f'  mean = {smoothed_stats["mean"]:.10g}')
	print(f'  median = {smoothed_stats["median"]:.10g}')
	print(f'  P95 = {smoothed_stats["p95"]:.10g}')
	print(f'  P99 = {smoothed_stats["p99"]:.10g}')
	print(f'  max = {smoothed_stats["maximum"]:.10g}')
	print(f'Maximum strain location: X={maximum_location[0]:.3f}, Y={maximum_location[1]:.3f}')
	print(f'P99 centroid: X={p99_centroid[0]:.3f}, Y={p99_centroid[1]:.3f}')
	print(f'Colour limits: vmin={vmin:.10g}, vmax={vmax:.10g}')
	print(f'Output files: {RAW_OUTPUT_PATH}, {SMOOTHED_OUTPUT_PATH}, {OVERLAY_OUTPUT_PATH}')


if __name__ == '__main__':
	main()