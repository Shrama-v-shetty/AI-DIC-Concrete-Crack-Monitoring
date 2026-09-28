from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from local_affine_strain_diagnostic import (
	MIN_NEIGHBORS,
	MIN_ZNCC,
	NEIGHBOR_RADIUS,
	local_affine_principal_strain,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE9_DIR = PROJECT_ROOT / 'Dataset' / 'Stage9_Full_Sequential_DIC'
THRESHOLD_PERCENTILES = (90, 95, 99)
POINT_SPACING = 10.0
VERY_SMALL_COMPONENT_RATIO = 0.10


def connected_components(points, strain, threshold):
	component_indices = np.flatnonzero(np.isfinite(strain) & (strain >= threshold))
	component_points = points[component_indices]
	if len(component_indices) == 0:
		return []

	adjacency_radius = np.sqrt(2.0) * POINT_SPACING + 1e-6
	pairs = cKDTree(component_points).query_pairs(adjacency_radius)
	parent = np.arange(len(component_indices), dtype=int)
	sizes = np.ones(len(component_indices), dtype=int)

	def find(index):
		while parent[index] != index:
			parent[index] = parent[parent[index]]
			index = parent[index]
		return index

	for left, right in pairs:
		left_root = find(left)
		right_root = find(right)
		if left_root != right_root:
			if sizes[left_root] < sizes[right_root]:
				left_root, right_root = right_root, left_root
			parent[right_root] = left_root
			sizes[left_root] += sizes[right_root]

	groups = {}
	for local_index, point_index in enumerate(component_indices):
		groups.setdefault(find(local_index), []).append(int(point_index))
	return sorted(groups.values(), key=len, reverse=True)


def describe_component(rank, component, points, strain):
	indices = np.asarray(component, dtype=int)
	centroid_x, centroid_y = points[indices].mean(axis=0)
	maximum_strain = float(np.max(strain[indices]))
	return (
		f'  Component {rank}: size={len(component):,}, '
		f'centroid=({centroid_x:.1f}, {centroid_y:.1f}), '
		f'max_strain={maximum_strain:.10g}'
	)


def diagnose_pair(previous_frame, target_frame):
	csv_path = STAGE9_DIR / f'dic_{previous_frame}_{target_frame}.csv'
	if not csv_path.is_file():
		raise FileNotFoundError(f'Stage 9 DIC CSV not found: {csv_path}')

	raw = pd.read_csv(csv_path)
	columns = ['X', 'Y', 'dx', 'dy', 'ZNCC']
	missing = [column for column in columns if column not in raw.columns]
	if missing:
		raise ValueError('DIC CSV is missing columns: ' + ', '.join(missing))

	data = raw[columns].apply(pd.to_numeric, errors='coerce')
	finite = np.isfinite(data[columns].to_numpy()).all(axis=1)
	data = data.loc[finite & (data['ZNCC'] >= MIN_ZNCC)].reset_index(drop=True)
	points = data[['X', 'Y']].to_numpy(float)
	dx = data['dx'].to_numpy(float)
	dy = data['dy'].to_numpy(float)
	strain = local_affine_principal_strain(points, dx, dy)
	valid = np.isfinite(strain)
	valid_values = strain[valid]
	if valid_values.size == 0:
		raise ValueError('No valid local-affine strain points were calculated.')

	global_maximum_index = int(np.nanargmax(strain))
	print(f'\nDIC pair: {previous_frame} -> {target_frame}')
	print(f'DIC points after finite and ZNCC >= {MIN_ZNCC:.2f}: {len(data):,}')
	print(f'Valid local-affine strain points: {int(valid.sum()):,}')
	print(
		f'Local fit: radius={NEIGHBOR_RADIUS:g} px, '
		f'minimum neighbors={MIN_NEIGHBORS}'
	)
	print(
		f'Component adjacency: 8-neighbor lattice, '
		f'radius={np.sqrt(2.0) * POINT_SPACING:.6f} px'
	)

	for percentile in THRESHOLD_PERCENTILES:
		threshold = float(np.percentile(valid_values, percentile))
		components = connected_components(points, strain, threshold)
		print(f'\n  P{percentile} threshold = {threshold:.10g}')
		print(f'  Connected components: {len(components):,}')
		for rank, component in enumerate(components[:10], start=1):
			print(describe_component(rank, component, points, strain))

		maximum_component = next(
			(component for component in components if global_maximum_index in component),
			None,
		)
		if maximum_component is None:
			print('  Global-maximum component: not present at this threshold')
			continue

		maximum_component_indices = np.asarray(maximum_component, dtype=int)
		maximum_centroid = points[maximum_component_indices].mean(axis=0)
		component_maximum_strain = float(np.max(strain[maximum_component_indices]))
		largest_component_size = len(components[0])
		ratio = len(maximum_component) / largest_component_size
		very_small = ratio < VERY_SMALL_COMPONENT_RATIO
		print(
			f'  Global-maximum component: size={len(maximum_component):,}, '
			f'centroid=({maximum_centroid[0]:.1f}, {maximum_centroid[1]:.1f})'
		)
		print(f'  Global-maximum component max strain: {component_maximum_strain:.10g}')
		print(f'  Largest component size: {largest_component_size:,}')
		print(
			f'  Very small (<{VERY_SMALL_COMPONENT_RATIO:.0%} of largest): '
			f'{"yes" if very_small else "no"}; size ratio={ratio:.1%}'
		)


def main():
	for previous_frame, target_frame in ((6429, 6430), (6399, 6400)):
		diagnose_pair(previous_frame, target_frame)


if __name__ == '__main__':
	main()