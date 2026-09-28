from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / 'Dataset'
STAGE9_DIR = DATASET_DIR / 'Stage9_Full_Sequential_DIC'
ROI_DIR = DATASET_DIR / 'BGSL_ROI'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'diagnostics'

MIN_ZNCC = 0.95
NEIGHBOR_RADIUS = 40.0
MIN_NEIGHBORS = 20
GRID_STEP = 10.0
VISUALIZATION_SUPPORT_RADIUS = 35.0
BOUNDARY_MARGIN = 40.0


def local_affine_principal_strain(points, dx, dy):
	"""Fit local displacement planes and calculate pointwise Green-Lagrange strain."""
	tree = cKDTree(points)
	strain = np.full(len(points), np.nan, dtype=float)

	for point_index, center in enumerate(points):
		neighbors = tree.query_ball_point(center, r=NEIGHBOR_RADIUS)
		if len(neighbors) < MIN_NEIGHBORS:
			continue

		neighbor_points = points[neighbors]
		local_coordinates = neighbor_points - center
		design = np.column_stack((
			np.ones(len(neighbors)),
			local_coordinates[:, 0],
			local_coordinates[:, 1],
		))
		if np.linalg.matrix_rank(design) < 3:
			continue

		coefficients_dx = np.linalg.lstsq(design, dx[neighbors], rcond=None)[0]
		coefficients_dy = np.linalg.lstsq(design, dy[neighbors], rcond=None)[0]
		du_dx, du_dy = coefficients_dx[1], coefficients_dx[2]
		dv_dx, dv_dy = coefficients_dy[1], coefficients_dy[2]

		deformation_gradient = np.array([
			[1.0 + du_dx, du_dy],
			[dv_dx, 1.0 + dv_dy],
		])
		green_lagrange = 0.5 * (
			deformation_gradient.T @ deformation_gradient - np.eye(2)
		)
		strain[point_index] = np.linalg.eigvalsh(green_lagrange)[-1]

	return strain


def interpolate_strain_for_visualization(points, strain):
	valid = np.isfinite(strain)
	valid_points = points[valid]
	valid_strain = strain[valid]
	if len(valid_points) < 3:
		raise ValueError('Too few valid local-affine strain points to interpolate.')

	grid_x_values = np.arange(
		valid_points[:, 0].min(), valid_points[:, 0].max() + GRID_STEP, GRID_STEP
	)
	grid_y_values = np.arange(
		valid_points[:, 1].min(), valid_points[:, 1].max() + GRID_STEP, GRID_STEP
	)
	grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
	grid_strain = griddata(
		valid_points, valid_strain, (grid_x, grid_y), method='linear'
	)
	nearest_distance, _ = cKDTree(valid_points).query(
		np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1
	)
	support = nearest_distance.reshape(grid_x.shape) <= VISUALIZATION_SUPPORT_RADIUS
	grid_strain[~support] = np.nan
	return grid_x, grid_y, grid_strain


def save_overlay(grid_x, grid_y, grid_strain, target_image, output_path, pair):
	values = grid_strain[np.isfinite(grid_strain)]
	if values.size == 0:
		raise ValueError('No valid visualization grid cells were produced.')
	vmin, vmax = np.percentile(values, [5, 99])
	if vmax <= vmin:
		vmax = float(np.nextafter(vmin, np.inf))

	cmap = plt.get_cmap('turbo').copy()
	cmap.set_bad((0, 0, 0, 0))
	height, width = target_image.shape[:2]
	fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
	ax.imshow(target_image, extent=[0, width, height, 0], origin='upper')
	heatmap = ax.pcolormesh(
		grid_x,
		grid_y,
		np.ma.masked_invalid(grid_strain),
		shading='nearest',
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		alpha=0.72,
		rasterized=True,
	)
	ax.set_xlim(0, width)
	ax.set_ylim(height, 0)
	ax.set_title(f'Local-affine maximum principal strain | {pair[0]} → {pair[1]}')
	ax.set_xlabel('ROI X (pixels)')
	ax.set_ylabel('ROI Y (pixels)')
	fig.colorbar(heatmap, ax=ax, label='Maximum Principal Green-Lagrange Strain')
	fig.savefig(output_path, dpi=180)
	plt.close(fig)
	return float(vmin), float(vmax), int(values.size)


def diagnose_pair(previous_frame, target_frame):
	csv_path = STAGE9_DIR / f'dic_{previous_frame}_{target_frame}.csv'
	image_path = ROI_DIR / f'BGSL-2-{target_frame}_0.tif'
	if not csv_path.is_file():
		raise FileNotFoundError(f'Stage 9 DIC CSV not found: {csv_path}')
	if not image_path.is_file():
		raise FileNotFoundError(f'Target ROI image not found: {image_path}')

	raw = pd.read_csv(csv_path)
	required = ['X', 'Y', 'dx', 'dy', 'ZNCC']
	missing = [column for column in required if column not in raw.columns]
	if missing:
		raise ValueError('DIC CSV is missing columns: ' + ', '.join(missing))

	data = raw[required].apply(pd.to_numeric, errors='coerce')
	finite = np.isfinite(data[required].to_numpy()).all(axis=1)
	data = data.loc[finite & (data['ZNCC'] >= MIN_ZNCC)].reset_index(drop=True)
	points = data[['X', 'Y']].to_numpy(float)
	dx = data['dx'].to_numpy(float)
	dy = data['dy'].to_numpy(float)
	point_strain = local_affine_principal_strain(points, dx, dy)
	valid_before_boundary = np.isfinite(point_strain)
	target_image = np.asarray(Image.open(image_path).convert('RGB'))
	roi_height, roi_width = target_image.shape[:2]
	inside_boundary = (
		(points[:, 0] >= BOUNDARY_MARGIN)
		& (points[:, 0] <= roi_width - BOUNDARY_MARGIN)
		& (points[:, 1] >= BOUNDARY_MARGIN)
		& (points[:, 1] <= roi_height - BOUNDARY_MARGIN)
	)
	point_strain[~inside_boundary] = np.nan
	valid_point_strain = np.isfinite(point_strain)
	if not np.any(valid_point_strain):
		raise ValueError('No DIC points had enough non-collinear neighbors for fitting.')

	strain_values = point_strain[valid_point_strain]
	valid_strain_points = points[valid_point_strain]
	maximum_index = int(np.argmax(strain_values))
	maximum_location = valid_strain_points[maximum_index]
	p99 = float(np.percentile(strain_values, 99))
	p99_points = valid_strain_points[strain_values >= p99]
	p99_centroid = p99_points.mean(axis=0)
	grid_x, grid_y, grid_strain = interpolate_strain_for_visualization(
		points, point_strain
	)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = OUTPUT_DIR / (
		f'local_affine_strain_{previous_frame}_{target_frame}_edge_margin{BOUNDARY_MARGIN:g}.png'
	)
	vmin, vmax, valid_grid_cells = save_overlay(
		grid_x,
		grid_y,
		grid_strain,
		target_image,
		output_path,
		(previous_frame, target_frame),
	)

	print(f'\nDIC pair: {previous_frame} → {target_frame}')
	print(f'DIC points (finite, ZNCC >= {MIN_ZNCC:.2f}): {len(data):,}')
	print(
		f'Valid strain points excluded by {BOUNDARY_MARGIN:g} px ROI margin '
		f'({roi_width}×{roi_height}): '
		f'{np.count_nonzero(valid_before_boundary & ~inside_boundary):,}'
	)
	print(f'Valid local-affine strain points: {len(strain_values):,}')
	print(f'Mean principal strain: {np.mean(strain_values):.10g}')
	print(f'Median principal strain: {np.median(strain_values):.10g}')
	print(f'P95 principal strain: {np.percentile(strain_values, 95):.10g}')
	print(f'P99 principal strain: {p99:.10g}')
	print(
		f'Maximum principal strain: {strain_values[maximum_index]:.10g} '
		f'at X={maximum_location[0]:.1f}, Y={maximum_location[1]:.1f}'
	)
	print(f'P99 centroid: X={p99_centroid[0]:.1f}, Y={p99_centroid[1]:.1f}')
	print(f'Valid visualization grid cells: {valid_grid_cells:,}')
	print(f'Colour limits (visualization P5/P99): {vmin:.10g} / {vmax:.10g}')
	print(f'Overlay: {output_path}')


def main():
	for previous_frame, target_frame in ((6429, 6430), (6399, 6400)):
		diagnose_pair(previous_frame, target_frame)


if __name__ == '__main__':
	main()