from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'Dataset' / 'Stage9_Full_Sequential_DIC' / 'dic_6429_6430.csv'
ROI_IMAGE_PATH = PROJECT_ROOT / 'Dataset' / 'BGSL_ROI' / 'BGSL-2-6429_0.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'diagnostics'
OVERLAY_PATH = OUTPUT_DIR / 'dic_6429_6430_displacement_heatmap.png'
FIELD_ONLY_PATH = OUTPUT_DIR / 'dic_6429_6430_displacement_field_only.png'
SUPPORT_RADIUS = 35.0
GRID_STEP = 5


def nearest_point_distances(grid_points, dic_points, chunk_size=256):
	"""Return exact Euclidean distance to measured points without a spatial tree."""
	distances = np.empty(len(grid_points), dtype=float)
	for start in range(0, len(grid_points), chunk_size):
		stop = min(start + chunk_size, len(grid_points))
		distances[start:stop] = cdist(
			grid_points[start:stop], dic_points
		).min(axis=1)
	return distances


def main():
	if not CSV_PATH.is_file():
		raise FileNotFoundError(f'Stage 9 CSV not found: {CSV_PATH}')
	if not ROI_IMAGE_PATH.is_file():
		raise FileNotFoundError(f'Frame 6429 ROI image not found: {ROI_IMAGE_PATH}')

	raw = pd.read_csv(CSV_PATH)
	required = {'X', 'Y', 'dx', 'dy'}
	missing = sorted(required.difference(raw.columns))
	if missing:
		raise ValueError('DIC CSV is missing required columns: ' + ', '.join(missing))

	data = raw.copy()
	for column in ('X', 'Y', 'dx', 'dy'):
		data[column] = pd.to_numeric(data[column], errors='coerce')
	valid_mask = np.isfinite(data[['X', 'Y', 'dx', 'dy']].to_numpy()).all(axis=1)
	data = data.loc[valid_mask].copy()
	if data.empty:
		raise ValueError('The DIC CSV contains no finite X, Y, dx, dy points.')

	data['magnitude'] = np.hypot(data['dx'], data['dy'])
	print(f'Number of points: {len(data):,} valid of {len(raw):,} CSV rows')
	for column in ('dx', 'dy', 'magnitude'):
		values = data[column].to_numpy(float)
		label = 'Displacement magnitude' if column == 'magnitude' else column
		print(
			f'{label} (pixels): min={np.min(values):.10g}, '
			f'max={np.max(values):.10g}, mean={np.mean(values):.10g}, '
			f'median={np.median(values):.10g}'
		)
		if column == 'magnitude':
			print(
				f'  P95={np.percentile(values, 95):.10g}, '
				f'P99={np.percentile(values, 99):.10g}'
			)

	if 'ZNCC' in data.columns:
		zncc = pd.to_numeric(data['ZNCC'], errors='coerce').to_numpy(float)
		zncc = zncc[np.isfinite(zncc)]
		if len(zncc):
			print(f'ZNCC: mean={np.mean(zncc):.10g}, min={np.min(zncc):.10g}')
		else:
			print('ZNCC: column exists but contains no finite values')

	image = np.asarray(Image.open(ROI_IMAGE_PATH).convert('RGB'))
	height, width = image.shape[:2]
	points = data[['X', 'Y']].to_numpy(float)
	magnitudes = data['magnitude'].to_numpy(float)
	if (
		np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
		or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)
	):
		raise ValueError(
			'DIC coordinates fall outside the frame 6429 ROI image. '
			'Check that this ROI image matches the Stage 9 coordinates.'
		)

	gx = np.arange(0, width, GRID_STEP, dtype=float)
	gy = np.arange(0, height, GRID_STEP, dtype=float)
	grid_x, grid_y = np.meshgrid(gx, gy)
	query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

	# Linear interpolation remains undefined outside the measured convex hull.
	field = griddata(points, magnitudes, (grid_x, grid_y), method='linear')

	# Also require each plotted cell to be within 35 px of an actual DIC point.
	support_distance = nearest_point_distances(query_points, points)
	supported = support_distance.reshape(grid_x.shape) <= SUPPORT_RADIUS
	field[~supported] = np.nan
	field = np.ma.masked_invalid(field)

	cmap = plt.get_cmap('cividis').copy()
	cmap.set_bad((0, 0, 0, 0))
	vmax = float(np.max(magnitudes))
	if vmax <= 0:
		vmax = 1e-12

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	extent = [0, width, height, 0]

	fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
	ax.imshow(image, extent=extent, origin='upper')
	heatmap = ax.imshow(
		field,
		extent=extent,
		origin='upper',
		cmap=cmap,
		vmin=0,
		vmax=vmax,
		alpha=0.78,
		interpolation='none',
	)
	ax.scatter(
		points[:, 0], points[:, 1],
		s=1.5, c='white', alpha=0.10, linewidths=0,
	)
	ax.set_title('Frame 6429 ROI with Pair 6429 → 6430 DIC Displacement Magnitude')
	ax.set_xlabel('ROI X (pixels)')
	ax.set_ylabel('ROI Y (pixels)')
	fig.colorbar(heatmap, ax=ax, label='Displacement magnitude (pixels)')
	fig.savefig(OVERLAY_PATH, dpi=180)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
	heatmap = ax.imshow(
		field,
		extent=extent,
		origin='upper',
		cmap=cmap,
		vmin=0,
		vmax=vmax,
		interpolation='none',
	)
	ax.scatter(
		points[:, 0], points[:, 1],
		s=1.5, c='black', alpha=0.12, linewidths=0,
	)
	ax.set_title('Pair 6429 → 6430 DIC Displacement Magnitude Field')
	ax.set_xlabel('ROI X (pixels)')
	ax.set_ylabel('ROI Y (pixels)')
	fig.colorbar(heatmap, ax=ax, label='Displacement magnitude (pixels)')
	fig.savefig(FIELD_ONLY_PATH, dpi=180)
	plt.close(fig)

	print(f'Frame 6429 ROI image: {ROI_IMAGE_PATH}')
	print(f'Heatmap overlay output: {OVERLAY_PATH}')
	print(f'Field-only output: {FIELD_ONLY_PATH}')
	print(
		f'Interpolation: linear griddata; unsupported/out-of-hull cells remain blank; '
		f'support radius={SUPPORT_RADIUS:g} px.'
	)


if __name__ == '__main__':
	main()
