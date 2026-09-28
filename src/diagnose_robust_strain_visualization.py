from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import (
	 gaussian_filter,
	 label,
	 median_filter,
	 distance_transform_edt,
)
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / 'Dataset'
STAGE9_DIR = DATASET_DIR / 'Stage9_Full_Sequential_DIC'
ROI_DIR = DATASET_DIR / 'BGSL_ROI'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'diagnostics'

MIN_ZNCC = 0.95
GRID_STEP = 10.0
SUPPORT_RADIUS = 35.0
DISPLACEMENT_SMOOTHING_SIGMA = 2.0
STRAIN_MEDIAN_SIZE = 3
SMALL_COMPONENT_AREA = 3
COMPONENT_THRESHOLD_PERCENTILE = 95.0


def current_strain_grid(previous_frame, target_frame):
	"""Reproduce the dashboard's current strain calculation without changes."""
	csv_path = STAGE9_DIR / f'dic_{previous_frame}_{target_frame}.csv'
	data = pd.read_csv(csv_path)
	required = ['X', 'Y', 'dx', 'dy', 'ZNCC']
	missing = [column for column in required if column not in data.columns]
	if missing:
		raise ValueError('Stage 9 CSV is missing columns: ' + ', '.join(missing))

	numeric = data[required].apply(pd.to_numeric, errors='coerce')
	finite = np.isfinite(numeric[required].to_numpy()).all(axis=1)
	valid_points = numeric.loc[finite & (numeric['ZNCC'] >= MIN_ZNCC)]
	if len(valid_points) < 3:
		raise ValueError(f'Not enough valid DIC points for {previous_frame}->{target_frame}.')

	points = valid_points[['X', 'Y']].to_numpy(float)
	grid_x_values = np.arange(points[:, 0].min(), points[:, 0].max() + GRID_STEP, GRID_STEP)
	grid_y_values = np.arange(points[:, 1].min(), points[:, 1].max() + GRID_STEP, GRID_STEP)
	grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
	dx_grid = griddata(points, valid_points['dx'].to_numpy(float), (grid_x, grid_y), method='linear')
	dy_grid = griddata(points, valid_points['dy'].to_numpy(float), (grid_x, grid_y), method='linear')

	nearest_distance, _ = cKDTree(points).query(
		np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1
	)
	support = nearest_distance.reshape(grid_x.shape) <= SUPPORT_RADIUS
	valid_grid = support & np.isfinite(dx_grid) & np.isfinite(dy_grid)
	dx_grid[~valid_grid] = np.nan
	dy_grid[~valid_grid] = np.nan

	support_weights = gaussian_filter(
		valid_grid.astype(float), sigma=DISPLACEMENT_SMOOTHING_SIGMA
	)
	smoothed_dx = gaussian_filter(
		np.where(valid_grid, dx_grid, 0.0), sigma=DISPLACEMENT_SMOOTHING_SIGMA
	)
	smoothed_dy = gaussian_filter(
		np.where(valid_grid, dy_grid, 0.0), sigma=DISPLACEMENT_SMOOTHING_SIGMA
	)
	dx_grid[valid_grid] = smoothed_dx[valid_grid] / support_weights[valid_grid]
	dy_grid[valid_grid] = smoothed_dy[valid_grid] / support_weights[valid_grid]
	dx_grid[~valid_grid] = np.nan
	dy_grid[~valid_grid] = np.nan

	du_dy, du_dx = np.gradient(dx_grid, grid_y_values, grid_x_values, edge_order=1)
	dv_dy, dv_dx = np.gradient(dy_grid, grid_y_values, grid_x_values, edge_order=1)
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
	return grid_x, grid_y, strain, valid_points


def count_components(mask):
	structure = np.ones((3, 3), dtype=bool)
	labels, count = label(mask, structure=structure)
	return labels, int(count)


def robust_spatial_strain(strain):
	valid_before = np.isfinite(strain)
	values_before = strain[valid_before]
	if values_before.size == 0:
		raise ValueError('No valid strain cells are available for visualization.')

	threshold = float(np.percentile(values_before, COMPONENT_THRESHOLD_PERCENTILE))
	high_strain_regions = valid_before & (strain >= threshold)
	labels, components_before = count_components(high_strain_regions)
	component_sizes = np.bincount(labels.ravel())
	remove_component = component_sizes < SMALL_COMPONENT_AREA
	remove_component[0] = False
	remove_cells = remove_component[labels] & high_strain_regions

	retained = strain.copy()
	retained[remove_cells] = np.nan
	valid_after_area_filter = np.isfinite(retained)
	filtered_labels, _ = count_components(high_strain_regions & ~remove_cells)
	components_after = int(np.unique(filtered_labels[filtered_labels > 0]).size)

	if not np.any(valid_after_area_filter):
		raise ValueError('Area filtering removed all valid strain cells.')
	nearest_indices = distance_transform_edt(
		~valid_after_area_filter, return_distances=False, return_indices=True
	)
	nearest_filled = retained[tuple(nearest_indices)]
	median_values = median_filter(nearest_filled, size=STRAIN_MEDIAN_SIZE, mode='nearest')
	displayed = np.full(strain.shape, np.nan, dtype=float)
	displayed[valid_after_area_filter] = median_values[valid_after_area_filter]

	final_values = displayed[np.isfinite(displayed)]
	maximum_index = int(np.nanargmax(displayed))
	maximum_row, maximum_column = np.unravel_index(maximum_index, displayed.shape)
	return displayed, {
		'valid_before': int(valid_before.sum()),
		'valid_after': int(np.isfinite(displayed).sum()),
		'components_before': components_before,
		'components_after': components_after,
		'component_threshold': threshold,
		'raw_p95': float(np.percentile(values_before, 95)),
		'raw_p99': float(np.percentile(values_before, 99)),
		'raw_max': float(np.max(values_before)),
		'filtered_p95': float(np.percentile(final_values, 95)),
		'filtered_p99': float(np.percentile(final_values, 99)),
		'filtered_max': float(np.max(final_values)),
		'vmin': float(np.percentile(final_values, 5)),
		'vmax': float(np.percentile(final_values, 99)),
		'max_row': int(maximum_row),
		'max_column': int(maximum_column),
	}


def save_overlay(field, grid_x, grid_y, target_image, stats, path, pair):
	height, width = target_image.shape[:2]
	cmap = plt.get_cmap('turbo').copy()
	cmap.set_bad((0, 0, 0, 0))
	fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
	ax.imshow(target_image, extent=[0, width, height, 0], origin='upper')
	heatmap = ax.pcolormesh(
		grid_x,
		grid_y,
		np.ma.masked_invalid(field),
		shading='nearest',
		cmap=cmap,
		vmin=stats['vmin'],
		vmax=max(stats['vmax'], stats['vmin'] + 1e-12),
		alpha=0.72,
		rasterized=True,
	)
	ax.set_xlim(0, width)
	ax.set_ylim(height, 0)
	ax.set_title(f'Robust maximum principal strain | {pair[0]} → {pair[1]}')
	ax.set_xlabel('ROI X (pixels)')
	ax.set_ylabel('ROI Y (pixels)')
	fig.colorbar(heatmap, ax=ax, label='Maximum Principal Green-Lagrange Strain')
	fig.savefig(path, dpi=180)
	plt.close(fig)


def diagnose_pair(previous_frame, target_frame):
	grid_x, grid_y, strain, valid_points = current_strain_grid(previous_frame, target_frame)
	displayed, stats = robust_spatial_strain(strain)
	image_path = ROI_DIR / f'BGSL-2-{target_frame}_0.tif'
	if not image_path.is_file():
		raise FileNotFoundError(f'Target ROI image not found: {image_path}')
	target_image = np.asarray(Image.open(image_path).convert('RGB'))
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = OUTPUT_DIR / f'strain_{previous_frame}_{target_frame}_robust_overlay.png'
	save_overlay(displayed, grid_x, grid_y, target_image, stats, output_path,
			 (previous_frame, target_frame))

	max_x = float(grid_x[stats['max_row'], stats['max_column']])
	max_y = float(grid_y[stats['max_row'], stats['max_column']])
	print(f'\nDIC pair: {previous_frame} → {target_frame}')
	print(f'Accepted DIC points: {len(valid_points):,}')
	print(
		f'Raw strain P95/P99/max: {stats["raw_p95"]:.10g} / '
		f'{stats["raw_p99"]:.10g} / {stats["raw_max"]:.10g}'
	)
	print(
		f'Filtered strain P95/P99/max: {stats["filtered_p95"]:.10g} / '
		f'{stats["filtered_p99"]:.10g} / {stats["filtered_max"]:.10g}'
	)
	print(f'Valid cells before filtering: {stats["valid_before"]:,}')
	print(f'Valid cells after filtering: {stats["valid_after"]:,}')
	print(
		f'Connected P{COMPONENT_THRESHOLD_PERCENTILE:g}+ strain components: '
		f'{stats["components_before"]:,} before, {stats["components_after"]:,} after '
		f'(small-component threshold={SMALL_COMPONENT_AREA} cells)'
	)
	print(f'Final colour limits (P5/P99): {stats["vmin"]:.10g} / {stats["vmax"]:.10g}')
	print(f'Final maximum X,Y: {max_x:.1f}, {max_y:.1f}')
	print(f'Overlay: {output_path}')


def main():
	for previous_frame, target_frame in ((6429, 6430), (6399, 6400)):
		diagnose_pair(previous_frame, target_frame)


if __name__ == '__main__':
	main()