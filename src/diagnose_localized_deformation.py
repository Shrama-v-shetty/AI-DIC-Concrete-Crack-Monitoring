from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from diagnose_strain_components import POINT_SPACING, connected_components
from local_affine_strain_diagnostic import (
    MIN_NEIGHBORS,
    MIN_ZNCC,
    NEIGHBOR_RADIUS,
    local_affine_principal_strain,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "Dataset"

STAGE9_DIR = DATASET_DIR / "Stage9_Full_Sequential_DIC"

ROI_DIR = DATASET_DIR / "BGSL_ROI"

OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"


# ============================================================
# STRAIN ANALYSIS SETTINGS
# ============================================================

THRESHOLD_PERCENTILES = (90, 95, 99)

MIN_VALID_POINTS_FOR_CLASSIFICATION = 1000

MIN_POINTS_PER_THRESHOLD = 20


# ============================================================
# LOCALIZATION SCORE SETTINGS
# ============================================================

COMPONENT_VALID_FRACTION_SCALE = 0.01

STRAIN_CONTRAST_SCALE = 0.50

LOCALIZATION_SCORE_WEIGHTS = {
    "dominance": 0.30,
    "maximum_membership": 0.15,
    "cross_threshold_consistency": 0.25,
    "strain_magnitude": 0.15,
    "spatial_compactness": 0.15,
}


# ============================================================
# LOCALIZED DEFORMATION CUT-OFFS
# ============================================================

MIN_LOCALIZATION_SCORE = 0.65

MIN_P95_COMPONENT_VALID_FRACTION = 0.01

MIN_P95_HIGH_STRAIN_SHARE = 0.40

MIN_P95_EXCESS_STRAIN_SHARE = 0.50

MIN_CROSS_THRESHOLD_CONSISTENCY = 0.70

MIN_SPATIAL_COMPACTNESS = 0.25


# ============================================================
# COMPONENT METRICS
# ============================================================

def component_metrics(
    component,
    points,
    strain,
    threshold,
    valid_count,
    threshold_point_count,
    largest_component_size,
    total_excess,
):
    """
    Calculate spatial and strain-related metrics
    for one connected high-strain component.
    """

    indices = np.asarray(component, dtype=int)

    component_points = points[indices]

    component_strain = strain[indices]

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid = component_points.mean(axis=0)

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox_width = float(
        np.ptp(component_points[:, 0]) + POINT_SPACING
    )

    bbox_height = float(
        np.ptp(component_points[:, 1]) + POINT_SPACING
    )

    bbox_cells = (
        (bbox_width / POINT_SPACING)
        * (bbox_height / POINT_SPACING)
    )

    # --------------------------------------------------------
    # Spatial compactness
    # --------------------------------------------------------

    spatial_compactness = min(
        float(len(indices) / bbox_cells),
        1.0,
    )

    # --------------------------------------------------------
    # Excess strain above threshold
    # --------------------------------------------------------

    component_excess = float(
        np.maximum(
            component_strain - threshold,
            0.0,
        ).sum()
    )

    if total_excess > 0:
        strain_concentration = (
            component_excess / total_excess
        )
    else:
        strain_concentration = 0.0

    # --------------------------------------------------------
    # Fraction of all valid points
    # --------------------------------------------------------

    size_valid_fraction = (
        len(indices) / valid_count
    )

    # --------------------------------------------------------
    # Fraction of all threshold points
    # --------------------------------------------------------

    size_high_strain_share = (
        len(indices) / threshold_point_count
    )

    # --------------------------------------------------------
    # Relative component size
    # --------------------------------------------------------

    size_relative_to_largest = (
        len(indices) / largest_component_size
    )

    # --------------------------------------------------------
    # Strain contrast
    # --------------------------------------------------------

    strain_contrast_score = float(
        min(
            max(
                float(
                    np.mean(component_strain)
                    / threshold
                    - 1.0
                ),
                0.0,
            )
            / STRAIN_CONTRAST_SCALE,
            1.0,
        )
    )

    return {
        "indices": indices,

        "size": int(len(indices)),

        "centroid": centroid,

        "maximum": float(
            np.max(component_strain)
        ),

        "mean": float(
            np.mean(component_strain)
        ),

        "bbox_width": bbox_width,

        "bbox_height": bbox_height,

        "size_valid_fraction": float(
            size_valid_fraction
        ),

        "size_high_strain_share": float(
            size_high_strain_share
        ),

        "size_relative_to_largest": float(
            size_relative_to_largest
        ),

        "strain_concentration": float(
            strain_concentration
        ),

        "strain_contrast_score": (
            strain_contrast_score
        ),

        "spatial_compactness": float(
            spatial_compactness
        ),
    }


# ============================================================
# CALCULATE CONNECTED COMPONENTS
# ============================================================

def calculate_threshold_components(
    points,
    strain,
    percentile,
):
    """
    Calculate connected high-strain components
    for a particular percentile threshold.
    """

    # --------------------------------------------------------
    # Only finite strain values
    # --------------------------------------------------------

    valid_values = strain[
        np.isfinite(strain)
    ]

    threshold = float(
        np.percentile(
            valid_values,
            percentile,
        )
    )

    # --------------------------------------------------------
    # Connected components above threshold
    # --------------------------------------------------------

    components = connected_components(
        points,
        strain,
        threshold,
    )

    threshold_point_count = int(
        np.count_nonzero(
            np.isfinite(strain)
            & (strain >= threshold)
        )
    )

    if not components:
        return (
            threshold,
            threshold_point_count,
            [],
        )

    # --------------------------------------------------------
    # Largest component
    # --------------------------------------------------------

    largest_component_size = max(
        map(len, components)
    )

    # --------------------------------------------------------
    # Total excess strain
    # --------------------------------------------------------

    threshold_values = valid_values[
        valid_values >= threshold
    ]

    total_excess = float(
        np.maximum(
            threshold_values - threshold,
            0.0,
        ).sum()
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    details = [
        component_metrics(
            component,
            points,
            strain,
            threshold,
            len(valid_values),
            threshold_point_count,
            largest_component_size,
            total_excess,
        )
        for component in components
    ]

    # Largest component first
    details.sort(
        key=lambda detail: detail["size"],
        reverse=True,
    )

    return (
        threshold,
        threshold_point_count,
        details,
    )


# ============================================================
# LOCALIZATION SCORE
# ============================================================

def localization_score(
    dominant_components,
    valid_count,
    global_maximum_index,
):
    """
    Calculate the overall localization score.
    """

    # --------------------------------------------------------
    # Component size scores
    # --------------------------------------------------------

    size_fraction_scores = [
        min(
            component["size_valid_fraction"]
            / COMPONENT_VALID_FRACTION_SCALE,
            1.0,
        )
        for component in dominant_components
    ]

    # --------------------------------------------------------
    # Dominance score
    # --------------------------------------------------------

    dominance_values = []

    for size_score, component in zip(
        size_fraction_scores,
        dominant_components,
    ):

        value = (
            0.50 * size_score
            + 0.25
            * component["size_high_strain_share"]
            + 0.25
            * component["strain_concentration"]
        )

        dominance_values.append(value)

    dominance_score = float(
        np.mean(dominance_values)
    )

    # --------------------------------------------------------
    # Maximum membership
    # --------------------------------------------------------

    maximum_membership_score = float(
        np.mean(
            [
                global_maximum_index
                in component["indices"]
                for component in dominant_components
            ]
        )
    )

    # --------------------------------------------------------
    # Adjacent threshold overlaps
    #
    # P90 vs P95
    # P95 vs P99
    # --------------------------------------------------------

    adjacent_overlaps = []

    for left, right in zip(
        dominant_components[:-1],
        dominant_components[1:],
    ):

        left_indices = set(
            left["indices"]
        )

        right_indices = set(
            right["indices"]
        )

        shared = len(
            left_indices & right_indices
        )

        smaller_component_size = min(
            len(left_indices),
            len(right_indices),
        )

        if smaller_component_size > 0:
            overlap = (
                shared
                / smaller_component_size
            )
        else:
            overlap = 0.0

        adjacent_overlaps.append(
            overlap
        )

    # --------------------------------------------------------
    # All-three overlap
    # --------------------------------------------------------

    all_indices = [
        set(component["indices"])
        for component in dominant_components
    ]

    shared_all = len(
        set.intersection(*all_indices)
    )

    all_overlap = (
        shared_all
        / min(
            map(len, all_indices)
        )
    )

    # --------------------------------------------------------
    # Cross-threshold consistency
    # --------------------------------------------------------

    cross_threshold_consistency = float(
        (
            np.mean(adjacent_overlaps)
            + all_overlap
        )
        / 2.0
    )

    # --------------------------------------------------------
    # Strain magnitude
    # --------------------------------------------------------

    strain_magnitude_score = float(
        np.mean(
            [
                component[
                    "strain_contrast_score"
                ]
                for component in dominant_components
            ]
        )
    )

    # --------------------------------------------------------
    # Spatial compactness
    # --------------------------------------------------------

    spatial_compactness_score = float(
        np.mean(
            [
                component[
                    "spatial_compactness"
                ]
                for component in dominant_components
            ]
        )
    )

    # --------------------------------------------------------
    # Score terms
    # --------------------------------------------------------

    score_terms = {
        "dominance": dominance_score,

        "maximum_membership":
            maximum_membership_score,

        "cross_threshold_consistency":
            cross_threshold_consistency,

        "strain_magnitude":
            strain_magnitude_score,

        "spatial_compactness":
            spatial_compactness_score,
    }

    # --------------------------------------------------------
    # Weighted total
    # --------------------------------------------------------

    total_score = sum(
        LOCALIZATION_SCORE_WEIGHTS[name]
        * score_terms[name]
        for name in LOCALIZATION_SCORE_WEIGHTS
    )

    return (
        total_score,
        score_terms,
        adjacent_overlaps,
        all_overlap,
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_localization(
    valid_count,
    threshold_counts,
    dominant_components,
    global_maximum_index,
    score_terms,
    total_score,
    adjacent_overlaps,
    all_overlap,
):
    """
    Classify deformation as:

        Localized deformation
        Distributed deformation
        Insufficient evidence
    """

    # --------------------------------------------------------
    # Insufficient evidence
    # --------------------------------------------------------

    if (
        valid_count
        < MIN_VALID_POINTS_FOR_CLASSIFICATION
        or any(
            count < MIN_POINTS_PER_THRESHOLD
            for count in threshold_counts
        )
        or not dominant_components
    ):
        return "Insufficient evidence"

    # --------------------------------------------------------
    # P95 dominant component
    # --------------------------------------------------------

    p95_component = dominant_components[1]

    # --------------------------------------------------------
    # Global maximum present in all thresholds?
    # --------------------------------------------------------

    maximum_in_all_thresholds = all(
        global_maximum_index
        in component["indices"]
        for component in dominant_components
    )

    # --------------------------------------------------------
    # Localization conditions
    # --------------------------------------------------------

    localized = (
        total_score
        >= MIN_LOCALIZATION_SCORE

        and maximum_in_all_thresholds

        and score_terms[
            "cross_threshold_consistency"
        ]
        >= MIN_CROSS_THRESHOLD_CONSISTENCY

        and p95_component[
            "size_valid_fraction"
        ]
        >= MIN_P95_COMPONENT_VALID_FRACTION

        and p95_component[
            "size_high_strain_share"
        ]
        >= MIN_P95_HIGH_STRAIN_SHARE

        and p95_component[
            "strain_concentration"
        ]
        >= MIN_P95_EXCESS_STRAIN_SHARE

        and p95_component[
            "spatial_compactness"
        ]
        >= MIN_SPATIAL_COMPACTNESS

        and all(
            overlap
            >= MIN_CROSS_THRESHOLD_CONSISTENCY
            for overlap in adjacent_overlaps
        )

        and all_overlap
        >= MIN_CROSS_THRESHOLD_CONSISTENCY
    )

    if localized:
        return "Localized deformation"

    return "Distributed deformation"


# ============================================================
# PRINT COMPONENT INFORMATION
# ============================================================

def print_dominant_component(
    percentile,
    threshold,
    threshold_point_count,
    detail,
):
    """
    Print information about the dominant component.
    """

    print(
        f"\n  P{percentile} threshold: "
        f"{threshold:.10g}"
    )

    print(
        f"  Points at/above threshold: "
        f"{threshold_point_count:,}"
    )

    centroid_x, centroid_y = (
        detail["centroid"]
    )

    print(
        f'  Connected components: '
        f'{detail["component_count"]:,}'
    )

    print(
        f'  Largest component size: '
        f'{detail["size"]:,}'
    )

    print(
        f'  Fraction of all valid strain points: '
        f'{detail["size_valid_fraction"]:.4%}'
    )

    print(
        f'  Fraction of largest component at this threshold: '
        f'{detail["size_relative_to_largest"]:.4%}'
    )

    print(
        f'  Fraction of largest dominant component across thresholds: '
        f'{detail["size_relative_to_largest_overall"]:.4%}'
    )

    print(
        f'  Fraction of threshold points in component: '
        f'{detail["size_high_strain_share"]:.4%}'
    )

    print(
        f'  Centroid: '
        f'({centroid_x:.1f}, {centroid_y:.1f})'
    )

    print(
        f'  Bounding box: '
        f'{detail["bbox_width"]:.1f} '
        f'x '
        f'{detail["bbox_height"]:.1f} px'
    )

    print(
        f'  Maximum strain: '
        f'{detail["maximum"]:.10g}'
    )

    print(
        f'  Mean strain: '
        f'{detail["mean"]:.10g}'
    )

    print(
        f'  Excess-strain concentration: '
        f'{detail["strain_concentration"]:.4f}'
    )

    print(
        f'  Spatial compactness: '
        f'{detail["spatial_compactness"]:.4f}'
    )

    print(
        f'  Global maximum belongs to component: '
        f'{"yes" if detail["contains_global_maximum"] else "no"}'
    )


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(
    points,
    dominant_components,
    target_image,
    output_path,
    pair,
    classification,
):
    """
    Draw dominant P90/P95/P99 components
    over the target ROI image.
    """

    height, width = (
        target_image.shape[:2]
    )

    fig, ax = plt.subplots(
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Background image
    # --------------------------------------------------------

    ax.imshow(
        target_image,
        extent=[
            0,
            width,
            height,
            0,
        ],
        origin="upper",
    )

    # --------------------------------------------------------
    # Visualization styles
    # --------------------------------------------------------

    styles = (
        (90, "#36d6c5", "o"),
        (95, "#ffe44d", "s"),
        (99, "#ff5c8a", "x"),
    )

    # --------------------------------------------------------
    # Draw each dominant component
    # --------------------------------------------------------

    for component, (
        percentile,
        color,
        marker,
    ) in zip(
        dominant_components,
        styles,
    ):

        component_points = points[
            component["indices"]
        ]

        if marker == "x":

            ax.scatter(
                component_points[:, 0],
                component_points[:, 1],
                s=34,
                marker=marker,
                c=color,
                linewidths=1.5,
                label=(
                    f"Dominant P{percentile} "
                    f"component"
                ),
            )

        else:

            ax.scatter(
                component_points[:, 0],
                component_points[:, 1],
                s=34,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                linewidths=1.2,
                label=(
                    f"Dominant P{percentile} "
                    f"component"
                ),
            )

    # --------------------------------------------------------
    # Axis settings
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        width,
    )

    ax.set_ylim(
        height,
        0,
    )

    ax.set_title(
        f"{pair[0]} → {pair[1]} | "
        f"{classification}"
    )

    ax.set_xlabel(
        "ROI X (pixels)"
    )

    ax.set_ylabel(
        "ROI Y (pixels)"
    )

    ax.legend(
        loc="upper right",
        framealpha=0.9,
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# ANALYZE ONE FRAME PAIR
# ============================================================

def diagnose_pair(
    previous_frame,
    target_frame,
):
    """
    Perform local-affine strain analysis and
    localization classification for one
    consecutive DIC pair.
    """

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    csv_path = (
        STAGE9_DIR
        / f"dic_{previous_frame}_{target_frame}.csv"
    )

    image_path = (
        ROI_DIR
        / f"BGSL-2-{target_frame}_0.tif"
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not csv_path.is_file():

        raise FileNotFoundError(
            f"Stage 9 DIC CSV not found: "
            f"{csv_path}"
        )

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Target ROI image not found: "
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    columns = [
        "X",
        "Y",
        "dx",
        "dy",
        "ZNCC",
    ]

    # --------------------------------------------------------
    # Read DIC data
    # --------------------------------------------------------

    raw = pd.read_csv(
        csv_path
    )

    missing = [
        column
        for column in columns
        if column not in raw.columns
    ]

    if missing:

        raise ValueError(
            "DIC CSV is missing columns: "
            + ", ".join(missing)
        )

    data = raw[
        columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    finite = np.isfinite(
        data[columns].to_numpy()
    ).all(axis=1)

    data = data.loc[
        finite
        & (
            data["ZNCC"]
            >= MIN_ZNCC
        )
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # DIC point coordinates
    # --------------------------------------------------------

    points = data[
        ["X", "Y"]
    ].to_numpy(
        float
    )

    # --------------------------------------------------------
    # Local affine principal strain
    # --------------------------------------------------------

    strain = (
        local_affine_principal_strain(
            points,
            data["dx"].to_numpy(float),
            data["dy"].to_numpy(float),
        )
    )

    # --------------------------------------------------------
    # Valid strain points
    # --------------------------------------------------------

    valid = np.isfinite(
        strain
    )

    valid_count = int(
        valid.sum()
    )

    if valid_count == 0:

        raise ValueError(
            "No valid local-affine strain "
            "points were calculated."
        )

    # --------------------------------------------------------
    # Global maximum strain point
    # --------------------------------------------------------

    global_maximum_index = int(
        np.nanargmax(strain)
    )

    # --------------------------------------------------------
    # Calculate dominant components
    # --------------------------------------------------------

    dominant_components = []

    threshold_counts = []

    for percentile in (
        THRESHOLD_PERCENTILES
    ):

        (
            threshold,
            threshold_count,
            components,
        ) = calculate_threshold_components(
            points,
            strain,
            percentile,
        )

        if not components:

            dominant_components = []

            break

        dominant = components[0]

        dominant[
            "component_count"
        ] = len(components)

        dominant[
            "contains_global_maximum"
        ] = (
            global_maximum_index
            in dominant["indices"]
        )

        dominant[
            "threshold"
        ] = threshold

        dominant_components.append(
            dominant
        )

        threshold_counts.append(
            threshold_count
        )

    # --------------------------------------------------------
    # Make sure all thresholds worked
    # --------------------------------------------------------

    if (
        len(dominant_components)
        != len(THRESHOLD_PERCENTILES)
    ):

        raise ValueError(
            "A threshold had no connected "
            "high-strain components."
        )

    # --------------------------------------------------------
    # Relative dominant component sizes
    # --------------------------------------------------------

    largest_dominant_size = max(
        component["size"]
        for component
        in dominant_components
    )

    for component in (
        dominant_components
    ):

        component[
            "size_relative_to_largest_overall"
        ] = (
            component["size"]
            / largest_dominant_size
        )

    # --------------------------------------------------------
    # Localization score
    # --------------------------------------------------------

    (
        total_score,
        score_terms,
        adjacent_overlaps,
        all_overlap,
    ) = localization_score(
        dominant_components,
        valid_count,
        global_maximum_index,
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = classify_localization(
        valid_count,
        threshold_counts,
        dominant_components,
        global_maximum_index,
        score_terms,
        total_score,
        adjacent_overlaps,
        all_overlap,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Pair: "
        f"{previous_frame} → "
        f"{target_frame}"
    )

    print(
        "=" * 70
    )

    print(
        f"DIC points after finite and "
        f"ZNCC >= {MIN_ZNCC:.2f}: "
        f"{len(data):,}"
    )

    print(
        f"Valid local-affine strain points: "
        f"{valid_count:,}"
    )

    print(
        f"Local affine: "
        f"radius={NEIGHBOR_RADIUS:g} px; "
        f"minimum neighbors={MIN_NEIGHBORS}"
    )

    print(
        "No ROI margin or strain-value "
        "filtering applied."
    )

    # --------------------------------------------------------
    # Print threshold information
    # --------------------------------------------------------

    for (
        percentile,
        count,
        component,
    ) in zip(
        THRESHOLD_PERCENTILES,
        threshold_counts,
        dominant_components,
    ):

        print_dominant_component(
            percentile,
            component["threshold"],
            count,
            component,
        )

    # --------------------------------------------------------
    # Cross-threshold consistency
    # --------------------------------------------------------

    print(
        "\n  Cross-threshold spatial consistency:"
    )

    for (
        lower,
        upper,
        overlap,
    ) in zip(
        THRESHOLD_PERCENTILES[:-1],
        THRESHOLD_PERCENTILES[1:],
        adjacent_overlaps,
    ):

        print(
            f"  P{lower}/P{upper} "
            f"smaller-component overlap: "
            f"{overlap:.1%}"
        )

    print(
        f"  P90/P95/P99 "
        f"shared-component overlap: "
        f"{all_overlap:.1%}"
    )

    # --------------------------------------------------------
    # Global maximum membership
    # --------------------------------------------------------

    maximum_membership_count = sum(
        component["contains_global_maximum"]
        for component in dominant_components
    )

    print(
        "  Global maximum belongs to "
        "dominant components: "
        f"{maximum_membership_count}/"
        f"{len(dominant_components)} thresholds"
    )

    # --------------------------------------------------------
    # Localization score
    # --------------------------------------------------------

    print(
        "\n  Localization score:"
    )

    print(
        "  Dominance = mean across thresholds of "
        "0.50*size-fraction score +"
    )

    print(
        "    0.25*high-threshold-point share + "
        "0.25*excess-strain share."
    )

    print(
        f"  Size-fraction score = "
        f"min(component size / valid points / "
        f"{COMPONENT_VALID_FRACTION_SCALE:.1%}, 1)."
    )

    print(
        f"  Strain-magnitude score = "
        f"mean of clamp((component mean / "
        f"threshold - 1) / "
        f"{STRAIN_CONTRAST_SCALE:.2f}, 0, 1)."
    )

    print(
        "  Compactness = component points / "
        "bounding-box lattice cells, capped at 1; "
        "consistency averages adjacent and "
        "all-three overlap fractions."
    )

    print(
        f"  Dominance="
        f"{score_terms['dominance']:.4f}; "
        f"maximum membership="
        f"{score_terms['maximum_membership']:.4f}; "
        f"cross-threshold consistency="
        f"{score_terms['cross_threshold_consistency']:.4f}; "
        f"strain magnitude="
        f"{score_terms['strain_magnitude']:.4f}; "
        f"spatial compactness="
        f"{score_terms['spatial_compactness']:.4f}"
    )

    # --------------------------------------------------------
    # Weight information
    # --------------------------------------------------------

    weight_text = "; ".join(
        f"{name}={weight:.0%}"
        for name, weight
        in LOCALIZATION_SCORE_WEIGHTS.items()
    )

    print(
        f"  Weighted score: "
        f"{weight_text}"
    )

    print(
        f"  Total localization score: "
        f"{total_score:.4f}"
    )

    # --------------------------------------------------------
    # Classification cutoffs
    # --------------------------------------------------------

    print(
        f"  Localized cutoffs: "
        f"score >= {MIN_LOCALIZATION_SCORE:.2f}; "
        f"all maxima inside dominant components; "
        f"consistency >= "
        f"{MIN_CROSS_THRESHOLD_CONSISTENCY:.0%}; "
        f"P95 component/valid >= "
        f"{MIN_P95_COMPONENT_VALID_FRACTION:.1%}; "
        f"P95 high-point share >= "
        f"{MIN_P95_HIGH_STRAIN_SHARE:.0%}; "
        f"P95 excess-strain share >= "
        f"{MIN_P95_EXCESS_STRAIN_SHARE:.0%}; "
        f"compactness >= "
        f"{MIN_SPATIAL_COMPACTNESS:.0%}"
    )

    # --------------------------------------------------------
    # Insufficient evidence settings
    # --------------------------------------------------------

    print(
        f"  Insufficient-evidence minimums: "
        f"valid points >= "
        f"{MIN_VALID_POINTS_FOR_CLASSIFICATION:,}; "
        f"each threshold has >= "
        f"{MIN_POINTS_PER_THRESHOLD} points."
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    print(
        f"\nClassification: "
        f"{classification}"
    )

    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"localized_deformation_"
            f"{previous_frame}_"
            f"{target_frame}.png"
        )
    )

    # --------------------------------------------------------
    # Load target ROI image
    # --------------------------------------------------------

    target_image = np.asarray(
        Image.open(
            image_path
        ).convert("RGB")
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    save_overlay(
        points,
        dominant_components,
        target_image,
        output_path,
        (
            previous_frame,
            target_frame,
        ),
        classification,
    )

    print(
        f"Overlay: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    test_pairs = (
        (1000, 1001),
        (1273, 1274),
        (2000, 2001),
        (3022, 3023),
        (6081, 6082),
        (6220, 6221),
        (6429, 6430),
        (6599, 6600),
    )

    for previous_frame, target_frame in test_pairs:

        diagnose_pair(
            previous_frame,
            target_frame,
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()