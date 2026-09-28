from pathlib import Path

import cv2
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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "Dataset"

STAGE9_DIR = DATASET_DIR / "Stage9_Full_Sequential_DIC"

ROI_DIR = DATASET_DIR / "BGSL_ROI"

OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"

CRACK_OUTPUT_DIR = OUTPUT_DIR / "crack_detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CRACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DIC / STRAIN SETTINGS
# ============================================================

THRESHOLD_PERCENTILES = (90, 95, 99)

MIN_VALID_POINTS_FOR_CLASSIFICATION = 1000

MIN_POINTS_PER_THRESHOLD = 20

COMPONENT_VALID_FRACTION_SCALE = 0.01

STRAIN_CONTRAST_SCALE = 0.50


# ============================================================
# LOCALIZATION SCORE
# ============================================================

LOCALIZATION_SCORE_WEIGHTS = {
    "dominance": 0.25,
    "maximum_membership": 0.10,
    "cross_threshold_consistency": 0.20,
    "strain_magnitude": 0.10,
    "spatial_compactness": 0.10,
    "crack_overlap": 0.25,
}


MIN_LOCALIZATION_SCORE = 0.65

MIN_P95_COMPONENT_VALID_FRACTION = 0.01

MIN_P95_HIGH_STRAIN_SHARE = 0.40

MIN_P95_EXCESS_STRAIN_SHARE = 0.50

MIN_CROSS_THRESHOLD_CONSISTENCY = 0.70

MIN_SPATIAL_COMPACTNESS = 0.25


# ============================================================
# CRACK DETECTION SETTINGS
# ============================================================

# Minimum connected component area for a crack candidate.
CRACK_MIN_AREA = 80

# Maximum number of crack components retained.
CRACK_MAX_COMPONENTS = 10

# Morphological kernel sizes.
CRACK_BLACKHAT_KERNEL = 31

CRACK_CLOSE_KERNEL = 21

# Crack response percentile.
CRACK_RESPONSE_PERCENTILE = 97.0

# Minimum crack component aspect ratio.
CRACK_MIN_ASPECT_RATIO = 2.0

# Minimum overlap between current and previous crack masks
# for temporal confirmation.
TEMPORAL_OVERLAP_THRESHOLD = 0.05

# Dilation radius used while checking temporal overlap.
TEMPORAL_DILATION_SIZE = 11

# Minimum fraction of a strain component that should overlap
# the detected crack region.
MIN_CRACK_COMPONENT_OVERLAP = 0.03

# Minimum crack confidence for using the crack information.
MIN_CRACK_CONFIDENCE = 0.45


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_gray_image(image_path):
    """
    Load ROI image as grayscale uint8.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    return image


def normalize_image(image):
    """
    Normalize illumination while preserving dark crack structures.
    """

    image = image.astype(np.float32)

    # Large Gaussian blur estimates local illumination.
    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=25,
        sigmaY=25,
    )

    # Avoid division by zero.
    background = np.maximum(background, 1.0)

    normalized = image / background

    # Convert to 8-bit.
    normalized = cv2.normalize(
        normalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return normalized.astype(np.uint8)


# ============================================================
# CRACK DETECTION
# ============================================================

def create_crack_response(image):
    """
    Detect dark elongated structures.

    A black-hat transform is used because a crack is typically
    darker than its surrounding concrete region.

    Several orientations are combined so the detector is not
    restricted to one crack direction.
    """

    normalized = normalize_image(image)

    # Black-hat detects dark structures relative to the local
    # surrounding intensity.
    kernel_size = CRACK_BLACKHAT_KERNEL

    kernel_square = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    blackhat_square = cv2.morphologyEx(
        normalized,
        cv2.MORPH_BLACKHAT,
        kernel_square,
    )

    # Directional kernels.
    line_horizontal = np.zeros(
        (15, 15),
        dtype=np.uint8,
    )
    line_horizontal[7, :] = 1

    line_vertical = np.zeros(
        (15, 15),
        dtype=np.uint8,
    )
    line_vertical[:, 7] = 1

    line_diag1 = np.eye(
        15,
        dtype=np.uint8,
    )

    line_diag2 = np.fliplr(
        line_diag1
    ).copy()

    responses = []

    for kernel in (
        line_horizontal,
        line_vertical,
        line_diag1,
        line_diag2,
    ):

        response = cv2.morphologyEx(
            normalized,
            cv2.MORPH_BLACKHAT,
            kernel,
        )

        responses.append(response.astype(np.float32))

    directional_response = np.maximum.reduce(
        responses
    )

    response = np.maximum(
        blackhat_square.astype(np.float32),
        directional_response,
    )

    response = cv2.GaussianBlur(
        response,
        (5, 5),
        0,
    )

    return response


def threshold_crack_response(response):
    """
    Convert crack response into a binary crack candidate mask.
    """

    finite_values = response[
        np.isfinite(response)
    ]

    if finite_values.size == 0:
        return np.zeros(
            response.shape,
            dtype=np.uint8,
        )

    threshold = np.percentile(
        finite_values,
        CRACK_RESPONSE_PERCENTILE,
    )

    mask = (
        response >= threshold
    ).astype(np.uint8) * 255

    # Connect nearby pixels belonging to the same crack.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            CRACK_CLOSE_KERNEL,
            CRACK_CLOSE_KERNEL,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    # Remove tiny isolated pixels.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small_kernel,
    )

    return mask


def component_aspect_ratio(x, y, w, h):
    """
    Aspect ratio based on the component bounding box.
    """

    long_side = max(w, h)

    short_side = max(
        min(w, h),
        1,
    )

    return float(long_side / short_side)


def detect_crack_components(image):
    """
    Detect candidate crack components.

    Returns:
        crack_mask
        crack_components
        crack_response
        crack_confidence
    """

    response = create_crack_response(image)

    binary_mask = threshold_crack_response(
        response
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    candidates = []

    for label in range(
        1,
        num_labels,
    ):

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < CRACK_MIN_AREA:
            continue

        aspect_ratio = component_aspect_ratio(
            x,
            y,
            w,
            h,
        )

        if aspect_ratio < CRACK_MIN_ASPECT_RATIO:
            continue

        component_mask = (
            labels == label
        )

        response_values = response[
            component_mask
        ]

        if response_values.size == 0:
            continue

        mean_response = float(
            np.mean(response_values)
        )

        max_response = float(
            np.max(response_values)
        )

        candidates.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "centroid": (
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                "mean_response": mean_response,
                "max_response": max_response,
                "mask": component_mask,
            }
        )

    # Sort by a combination of area and response.
    candidates.sort(
        key=lambda item: (
            item["area"]
            * item["mean_response"]
        ),
        reverse=True,
    )

    candidates = candidates[
        :CRACK_MAX_COMPONENTS
    ]

    crack_mask = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    for component in candidates:
        crack_mask[
            component["mask"]
        ] = 255

    # Crack confidence.
    if not candidates:
        confidence = 0.0

    else:

        largest = candidates[0]

        aspect_score = min(
            largest["aspect_ratio"] / 8.0,
            1.0,
        )

        area_score = min(
            largest["area"] / 2000.0,
            1.0,
        )

        response_score = min(
            largest["mean_response"] / 40.0,
            1.0,
        )

        confidence = float(
            0.35 * aspect_score
            + 0.25 * area_score
            + 0.40 * response_score
        )

        confidence = min(
            max(confidence, 0.0),
            1.0,
        )

    return (
        crack_mask,
        candidates,
        response,
        confidence,
    )


# ============================================================
# TEMPORAL CRACK DETECTION
# ============================================================

def calculate_temporal_overlap(
    current_mask,
    previous_mask,
):
    """
    Check whether the same crack-like region appears
    in consecutive images.
    """

    if (
        current_mask is None
        or previous_mask is None
    ):
        return 0.0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            TEMPORAL_DILATION_SIZE,
            TEMPORAL_DILATION_SIZE,
        ),
    )

    previous_dilated = cv2.dilate(
        previous_mask,
        kernel,
    )

    current_binary = (
        current_mask > 0
    )

    previous_binary = (
        previous_dilated > 0
    )

    current_count = int(
        np.count_nonzero(
            current_binary
        )
    )

    if current_count == 0:
        return 0.0

    overlap = np.count_nonzero(
        current_binary
        & previous_binary
    )

    return float(
        overlap / current_count
    )


def detect_crack_region(
    previous_image,
    target_image,
):
    """
    Detect crack candidates in both frames and combine
    spatial and temporal evidence.
    """

    (
        previous_mask,
        previous_components,
        previous_response,
        previous_confidence,
    ) = detect_crack_components(
        previous_image
    )

    (
        target_mask,
        target_components,
        target_response,
        target_confidence,
    ) = detect_crack_components(
        target_image
    )

    temporal_overlap = calculate_temporal_overlap(
        target_mask,
        previous_mask,
    )

    # --------------------------------------------------------
    # Temporal confidence
    # --------------------------------------------------------

    temporal_score = min(
        temporal_overlap
        / max(
            TEMPORAL_OVERLAP_THRESHOLD,
            1e-9,
        ),
        1.0,
    )

    crack_confidence = float(
        0.55 * target_confidence
        + 0.25 * previous_confidence
        + 0.20 * temporal_score
    )

    crack_confidence = min(
        max(
            crack_confidence,
            0.0,
        ),
        1.0,
    )

    # --------------------------------------------------------
    # If target detection is stronger, use target.
    # Otherwise combine target and previous.
    # --------------------------------------------------------

    final_mask = target_mask.copy()

    if (
        temporal_overlap
        >= TEMPORAL_OVERLAP_THRESHOLD
    ):

        final_mask = cv2.bitwise_or(
            target_mask,
            previous_mask,
        )

    return {
        "mask": final_mask,
        "target_mask": target_mask,
        "previous_mask": previous_mask,
        "target_components": target_components,
        "previous_components": previous_components,
        "target_response": target_response,
        "previous_response": previous_response,
        "target_confidence": target_confidence,
        "previous_confidence": previous_confidence,
        "temporal_overlap": temporal_overlap,
        "confidence": crack_confidence,
    }


# ============================================================
# CRACK MASK → DIC POINT OVERLAP
# ============================================================

def points_inside_mask(
    points,
    mask,
):
    """
    Determine which DIC points lie inside the crack mask.
    """

    height, width = mask.shape[:2]

    x = np.rint(
        points[:, 0]
    ).astype(int)

    y = np.rint(
        points[:, 1]
    ).astype(int)

    valid_coordinates = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    inside = np.zeros(
        len(points),
        dtype=bool,
    )

    inside[valid_coordinates] = (
        mask[
            y[valid_coordinates],
            x[valid_coordinates]
        ]
        > 0
    )

    return inside


def component_crack_overlap(
    component,
    crack_point_mask,
):
    """
    Fraction of component DIC points that overlap
    the image-detected crack region.
    """

    indices = component["indices"]

    if len(indices) == 0:
        return 0.0

    overlap_count = int(
        np.count_nonzero(
            crack_point_mask[indices]
        )
    )

    return float(
        overlap_count / len(indices)
    )


def find_crack_related_component(
    dominant_components,
    crack_point_mask,
):
    """
    Find which dominant strain component has the strongest
    relationship with the image-detected crack.
    """

    best_component = None

    best_overlap = 0.0

    for component in dominant_components:

        overlap = component_crack_overlap(
            component,
            crack_point_mask,
        )

        component[
            "crack_overlap"
        ] = overlap

        if overlap > best_overlap:

            best_overlap = overlap

            best_component = component

    return (
        best_component,
        best_overlap,
    )


# ============================================================
# STRAIN COMPONENT METRICS
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

    indices = np.asarray(
        component,
        dtype=int,
    )

    component_points = points[
        indices
    ]

    component_strain = strain[
        indices
    ]

    centroid = (
        component_points.mean(
            axis=0
        )
    )

    bbox_width = float(
        np.ptp(
            component_points[:, 0]
        )
        + POINT_SPACING
    )

    bbox_height = float(
        np.ptp(
            component_points[:, 1]
        )
        + POINT_SPACING
    )

    bbox_cells = (
        bbox_width
        / POINT_SPACING
    ) * (
        bbox_height
        / POINT_SPACING
    )

    spatial_compactness = min(
        float(
            len(indices)
            / bbox_cells
        ),
        1.0,
    )

    component_excess = float(
        np.maximum(
            component_strain
            - threshold,
            0.0,
        ).sum()
    )

    strain_concentration = (
        component_excess
        / total_excess
        if total_excess > 0
        else 0.0
    )

    size_valid_fraction = (
        len(indices)
        / valid_count
    )

    size_high_strain_share = (
        len(indices)
        / threshold_point_count
        if threshold_point_count > 0
        else 0.0
    )

    return {
        "indices": indices,
        "size": int(
            len(indices)
        ),
        "centroid": centroid,
        "maximum": float(
            np.max(
                component_strain
            )
        ),
        "mean": float(
            np.mean(
                component_strain
            )
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
            len(indices)
            / largest_component_size
        ),
        "strain_concentration": float(
            strain_concentration
        ),
        "strain_contrast_score": float(
            min(
                max(
                    float(
                        np.mean(
                            component_strain
                        )
                        / threshold
                        - 1.0
                    ),
                    0.0,
                )
                / STRAIN_CONTRAST_SCALE,
                1.0,
            )
        ),
        "spatial_compactness": float(
            spatial_compactness
        ),
    }


def calculate_threshold_components(
    points,
    strain,
    percentile,
):

    valid_values = strain[
        np.isfinite(strain)
    ]

    threshold = float(
        np.percentile(
            valid_values,
            percentile,
        )
    )

    components = connected_components(
        points,
        strain,
        threshold,
    )

    threshold_point_count = int(
        np.count_nonzero(
            np.isfinite(strain)
            & (
                strain
                >= threshold
            )
        )
    )

    if not components:
        return (
            threshold,
            threshold_point_count,
            [],
        )

    largest_component_size = max(
        map(
            len,
            components,
        )
    )

    total_excess = float(
        np.maximum(
            valid_values[
                valid_values
                >= threshold
            ]
            - threshold,
            0.0,
        ).sum()
    )

    details = []

    for component in components:

        details.append(
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
        )

    details.sort(
        key=lambda detail: detail[
            "size"
        ],
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

    size_fraction_scores = [
        min(
            component[
                "size_valid_fraction"
            ]
            / COMPONENT_VALID_FRACTION_SCALE,
            1.0,
        )
        for component
        in dominant_components
    ]

    dominance_score = float(
        np.mean(
            [
                0.5 * size_score
                + 0.25
                * component[
                    "size_high_strain_share"
                ]
                + 0.25
                * component[
                    "strain_concentration"
                ]
                for size_score, component
                in zip(
                    size_fraction_scores,
                    dominant_components,
                )
            ]
        )
    )

    maximum_membership_score = float(
        np.mean(
            [
                global_maximum_index
                in component[
                    "indices"
                ]
                for component
                in dominant_components
            ]
        )
    )

    # --------------------------------------------------------
    # Cross-threshold overlap
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
            left_indices
            & right_indices
        )

        minimum_size = min(
            len(left_indices),
            len(right_indices),
        )

        if minimum_size == 0:
            overlap = 0.0
        else:
            overlap = (
                shared
                / minimum_size
            )

        adjacent_overlaps.append(
            overlap
        )

    all_indices = [
        set(
            component[
                "indices"
            ]
        )
        for component
        in dominant_components
    ]

    shared_all = len(
        set.intersection(
            *all_indices
        )
    )

    all_overlap = (
        shared_all
        / min(
            map(
                len,
                all_indices,
            )
        )
    )

    cross_threshold_consistency = float(
        (
            np.mean(
                adjacent_overlaps
            )
            + all_overlap
        )
        / 2.0
    )

    strain_magnitude_score = float(
        np.mean(
            [
                component[
                    "strain_contrast_score"
                ]
                for component
                in dominant_components
            ]
        )
    )

    spatial_compactness_score = float(
        np.mean(
            [
                component[
                    "spatial_compactness"
                ]
                for component
                in dominant_components
            ]
        )
    )

    crack_overlap_score = float(
        np.mean(
            [
                component.get(
                    "crack_overlap",
                    0.0,
                )
                for component
                in dominant_components
            ]
        )
    )

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
        "crack_overlap":
            min(
                crack_overlap_score
                / 0.25,
                1.0,
            ),
    }

    total_score = sum(
        LOCALIZATION_SCORE_WEIGHTS[
            name
        ]
        * score_terms[name]
        for name
        in LOCALIZATION_SCORE_WEIGHTS
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
    crack_confidence,
):

    if (
        valid_count
        < MIN_VALID_POINTS_FOR_CLASSIFICATION
        or any(
            count
            < MIN_POINTS_PER_THRESHOLD
            for count
            in threshold_counts
        )
        or not dominant_components
    ):

        return "Insufficient evidence"

    p95_component = (
        dominant_components[1]
    )

    maximum_in_all_thresholds = all(
        global_maximum_index
        in component["indices"]
        for component
        in dominant_components
    )

    # --------------------------------------------------------
    # Original DIC localization evidence
    # --------------------------------------------------------

    dic_localized = (
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
            for overlap
            in adjacent_overlaps
        )
        and all_overlap
        >= MIN_CROSS_THRESHOLD_CONSISTENCY
    )

    # --------------------------------------------------------
    # Crack evidence
    # --------------------------------------------------------

    crack_related = (
        crack_confidence
        >= MIN_CRACK_CONFIDENCE
        and score_terms[
            "crack_overlap"
        ]
        > 0.0
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    if (
        dic_localized
        and crack_related
    ):

        return (
            "Localized crack-related deformation"
        )

    if dic_localized:

        return (
            "Localized deformation"
        )

    if crack_related:

        return (
            "Crack-region deformation"
        )

    return (
        "Distributed deformation"
    )


# ============================================================
# PRINT FUNCTIONS
# ============================================================

def print_dominant_component(
    percentile,
    threshold,
    threshold_point_count,
    detail,
):

    print(
        f"\n  P{percentile} threshold: "
        f"{threshold:.10g}"
    )

    print(
        "  Points at/above threshold: "
        f"{threshold_point_count:,}"
    )

    centroid_x, centroid_y = (
        detail["centroid"]
    )

    print(
        "  Connected components: "
        f"{detail['component_count']:,}"
    )

    print(
        "  Largest component size: "
        f"{detail['size']:,}"
    )

    print(
        "  Fraction of all valid strain points: "
        f"{detail['size_valid_fraction']:.4%}"
    )

    print(
        "  Fraction of threshold points in component: "
        f"{detail['size_high_strain_share']:.4%}"
    )

    print(
        f"  Centroid: "
        f"({centroid_x:.1f}, "
        f"{centroid_y:.1f})"
    )

    print(
        "  Bounding box: "
        f"{detail['bbox_width']:.1f} x "
        f"{detail['bbox_height']:.1f} px"
    )

    print(
        "  Maximum strain: "
        f"{detail['maximum']:.10g}"
    )

    print(
        "  Mean strain: "
        f"{detail['mean']:.10g}"
    )

    print(
        "  Excess-strain concentration: "
        f"{detail['strain_concentration']:.4f}"
    )

    print(
        "  Spatial compactness: "
        f"{detail['spatial_compactness']:.4f}"
    )

    print(
        "  Crack overlap: "
        f"{detail.get('crack_overlap', 0.0):.4%}"
    )

    print(
        "  Global maximum belongs to component: "
        f"{'yes' if detail['contains_global_maximum'] else 'no'}"
    )


# ============================================================
# CRACK COMPONENT PRINTING
# ============================================================

def print_crack_detection(
    crack_info,
):

    print(
        "\n  Image-based crack detection:"
    )

    print(
        "  Target crack confidence: "
        f"{crack_info['target_confidence']:.4f}"
    )

    print(
        "  Previous crack confidence: "
        f"{crack_info['previous_confidence']:.4f}"
    )

    print(
        "  Temporal overlap: "
        f"{crack_info['temporal_overlap']:.4%}"
    )

    print(
        "  Final crack confidence: "
        f"{crack_info['confidence']:.4f}"
    )

    print(
        "  Target crack candidates: "
        f"{len(crack_info['target_components'])}"
    )

    for index, component in enumerate(
        crack_info[
            "target_components"
        ],
        start=1,
    ):

        cx, cy = component[
            "centroid"
        ]

        print(
            f"    Candidate {index}: "
            f"area={component['area']}, "
            f"centroid=({cx:.1f}, {cy:.1f}), "
            f"bbox={component['width']}x"
            f"{component['height']}, "
            f"aspect={component['aspect_ratio']:.2f}"
        )


# ============================================================
# OVERLAY
# ============================================================

def save_overlay(
    points,
    dominant_components,
    target_image,
    crack_mask,
    crack_components,
    output_path,
    pair,
    classification,
):

    height, width = (
        target_image.shape[:2]
    )

    fig, ax = plt.subplots(
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    ax.imshow(
        target_image,
        extent=[
            0,
            width,
            height,
            0,
        ],
        origin="upper",
        cmap="gray",
    )

    # --------------------------------------------------------
    # Draw crack mask
    # --------------------------------------------------------

    if np.any(
        crack_mask > 0
    ):

        crack_overlay = np.zeros(
            (
                height,
                width,
                4,
            ),
            dtype=float,
        )

        crack_overlay[
            crack_mask > 0
        ] = (
            1.0,
            0.25,
            0.0,
            0.35,
        )

        ax.imshow(
            crack_overlay,
            extent=[
                0,
                width,
                height,
                0,
            ],
            origin="upper",
        )

    # --------------------------------------------------------
    # Crack component boundaries
    # --------------------------------------------------------

    for component in crack_components:

        x = component["x"]
        y = component["y"]
        w = component["width"]
        h = component["height"]

        rectangle = plt.Rectangle(
            (
                x,
                y,
            ),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )

        ax.add_patch(
            rectangle
        )

    # --------------------------------------------------------
    # DIC strain components
    # --------------------------------------------------------

    styles = (
        (
            90,
            "#36d6c5",
            "o",
        ),
        (
            95,
            "#ffe44d",
            "s",
        ),
        (
            99,
            "#ff5c8a",
            "x",
        ),
    )

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
                    "component"
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
                    "component"
                ),
            )

    # --------------------------------------------------------
    # Labels
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

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# CRACK-ONLY OVERLAY
# ============================================================

def save_crack_only_overlay(
    target_image,
    crack_mask,
    crack_components,
    output_path,
    pair,
):

    height, width = (
        target_image.shape[:2]
    )

    fig, ax = plt.subplots(
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    ax.imshow(
        target_image,
        cmap="gray",
        origin="upper",
    )

    # Crack mask.
    crack_overlay = np.zeros(
        (
            height,
            width,
            4,
        ),
        dtype=float,
    )

    crack_overlay[
        crack_mask > 0
    ] = (
        1.0,
        0.15,
        0.0,
        0.45,
    )

    ax.imshow(
        crack_overlay,
        origin="upper",
    )

    for component in crack_components:

        x = component["x"]
        y = component["y"]
        w = component["width"]
        h = component["height"]

        rectangle = plt.Rectangle(
            (
                x,
                y,
            ),
            w,
            h,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )

        ax.add_patch(
            rectangle
        )

        cx, cy = component[
            "centroid"
        ]

        ax.text(
            cx,
            cy,
            "crack",
            color="red",
            fontsize=10,
            weight="bold",
        )

    ax.set_title(
        f"Image-based crack detection | "
        f"{pair[0]} → {pair[1]}"
    )

    ax.set_xlabel(
        "ROI X (pixels)"
    )

    ax.set_ylabel(
        "ROI Y (pixels)"
    )

    ax.set_xlim(
        0,
        width,
    )

    ax.set_ylim(
        height,
        0,
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def diagnose_pair(
    previous_frame,
    target_frame,
):

    csv_path = (
        STAGE9_DIR
        / f"dic_{previous_frame}_{target_frame}.csv"
    )

    previous_image_path = (
        ROI_DIR
        / f"BGSL-2-{previous_frame}_0.tif"
    )

    target_image_path = (
        ROI_DIR
        / f"BGSL-2-{target_frame}_0.tif"
    )

    if not csv_path.is_file():

        raise FileNotFoundError(
            f"Stage 9 DIC CSV not found: "
            f"{csv_path}"
        )

    if not previous_image_path.is_file():

        raise FileNotFoundError(
            f"Previous ROI image not found: "
            f"{previous_image_path}"
        )

    if not target_image_path.is_file():

        raise FileNotFoundError(
            f"Target ROI image not found: "
            f"{target_image_path}"
        )

    # ========================================================
    # LOAD DIC
    # ========================================================

    columns = [
        "X",
        "Y",
        "dx",
        "dy",
        "ZNCC",
    ]

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

    finite = np.isfinite(
        data[
            columns
        ].to_numpy()
    ).all(
        axis=1
    )

    data = data.loc[
        finite
        & (
            data["ZNCC"]
            >= MIN_ZNCC
        )
    ].reset_index(
        drop=True
    )

    points = data[
        [
            "X",
            "Y",
        ]
    ].to_numpy(
        float
    )

    # ========================================================
    # LOCAL AFFINE STRAIN
    # ========================================================

    strain = (
        local_affine_principal_strain(
            points,
            data["dx"].to_numpy(
                float
            ),
            data["dy"].to_numpy(
                float
            ),
        )
    )

    valid = np.isfinite(
        strain
    )

    valid_count = int(
        valid.sum()
    )

    if valid_count == 0:

        raise ValueError(
            "No valid local-affine "
            "strain points were calculated."
        )

    global_maximum_index = int(
        np.nanargmax(
            strain
        )
    )

    # ========================================================
    # IMAGE-BASED CRACK DETECTION
    # ========================================================

    previous_image = (
        load_gray_image(
            previous_image_path
        )
    )

    target_image = (
        load_gray_image(
            target_image_path
        )
    )

    crack_info = (
        detect_crack_region(
            previous_image,
            target_image,
        )
    )

    crack_mask = crack_info[
        "mask"
    ]

    crack_components = crack_info[
        "target_components"
    ]

    crack_confidence = crack_info[
        "confidence"
    ]

    # ========================================================
    # MAP CRACK MASK TO DIC POINTS
    # ========================================================

    crack_point_mask = (
        points_inside_mask(
            points,
            crack_mask,
        )
    )

    crack_dic_points = int(
        crack_point_mask.sum()
    )

    # ========================================================
    # DIC COMPONENTS
    # ========================================================

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

        dominant = components[
            0
        ]

        dominant[
            "component_count"
        ] = len(
            components
        )

        dominant[
            "contains_global_maximum"
        ] = (
            global_maximum_index
            in dominant[
                "indices"
            ]
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

    if (
        len(
            dominant_components
        )
        != len(
            THRESHOLD_PERCENTILES
        )
    ):

        raise ValueError(
            "A threshold had no connected "
            "high-strain components."
        )

    # ========================================================
    # ADD CRACK OVERLAP TO EACH COMPONENT
    # ========================================================

    for component in (
        dominant_components
    ):

        overlap = (
            component_crack_overlap(
                component,
                crack_point_mask,
            )
        )

        component[
            "crack_overlap"
        ] = overlap

    (
        crack_related_component,
        crack_related_overlap,
    ) = find_crack_related_component(
        dominant_components,
        crack_point_mask,
    )

    # ========================================================
    # CROSS-THRESHOLD METRICS
    # ========================================================

    largest_dominant_size = max(
        component[
            "size"
        ]
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

    # ========================================================
    # LOCALIZATION SCORE
    # ========================================================

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

    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    classification = (
        classify_localization(
            valid_count,
            threshold_counts,
            dominant_components,
            global_maximum_index,
            score_terms,
            total_score,
            adjacent_overlaps,
            all_overlap,
            crack_confidence,
        )
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
        "DIC points after finite and "
        f"ZNCC >= {MIN_ZNCC:.2f}: "
        f"{len(data):,}"
    )

    print(
        "Valid local-affine strain points: "
        f"{valid_count:,}"
    )

    print(
        "Local affine: "
        f"radius={NEIGHBOR_RADIUS:g} px; "
        f"minimum neighbors={MIN_NEIGHBORS}"
    )

    print(
        "\nImage-based crack detection:"
    )

    print(
        "  Target image: "
        f"{target_image_path.name}"
    )

    print(
        "  Crack candidate components: "
        f"{len(crack_components)}"
    )

    print(
        "  DIC points inside crack mask: "
        f"{crack_dic_points:,}"
    )

    print(
        "  Target crack confidence: "
        f"{crack_info['target_confidence']:.4f}"
    )

    print(
        "  Previous crack confidence: "
        f"{crack_info['previous_confidence']:.4f}"
    )

    print(
        "  Temporal crack overlap: "
        f"{crack_info['temporal_overlap']:.4%}"
    )

    print(
        "  Final crack confidence: "
        f"{crack_confidence:.4f}"
    )

    if crack_related_component is not None:

        cx, cy = (
            crack_related_component[
                "centroid"
            ]
        )

        print(
            "  Best DIC/crack component: "
            f"centroid=({cx:.1f}, {cy:.1f})"
        )

        print(
            "  Best DIC/crack overlap: "
            f"{crack_related_overlap:.4%}"
        )

    else:

        print(
            "  No DIC component overlaps "
            "the detected crack region."
        )

    # ========================================================
    # PRINT STRAIN COMPONENTS
    # ========================================================

    print(
        "\nDIC strain components:"
    )

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
            component[
                "threshold"
            ],
            count,
            component,
        )

    # ========================================================
    # FIXED SYNTAX FOR GLOBAL MAXIMUM
    # ========================================================

    maximum_membership_count = sum(
        component[
            "contains_global_maximum"
        ]
        for component
        in dominant_components
    )

    print(
        "\n  Global maximum belongs to "
        "dominant components: "
        f"{maximum_membership_count}/"
        f"{len(dominant_components)} "
        "thresholds"
    )

    # ========================================================
    # CROSS THRESHOLD
    # ========================================================

    print(
        "\nCross-threshold spatial consistency:"
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
            f"component overlap: "
            f"{overlap:.1%}"
        )

    print(
        "  P90/P95/P99 shared-component "
        f"overlap: {all_overlap:.1%}"
    )

    # ========================================================
    # LOCALIZATION SCORE
    # ========================================================

    print(
        "\nLocalization score:"
    )

    print(
        f"  Dominance = "
        f"{score_terms['dominance']:.4f}"
    )

    print(
        f"  Maximum membership = "
        f"{score_terms['maximum_membership']:.4f}"
    )

    print(
        f"  Cross-threshold consistency = "
        f"{score_terms['cross_threshold_consistency']:.4f}"
    )

    print(
        f"  Strain magnitude = "
        f"{score_terms['strain_magnitude']:.4f}"
    )

    print(
        f"  Spatial compactness = "
        f"{score_terms['spatial_compactness']:.4f}"
    )

    print(
        f"  Crack overlap = "
        f"{score_terms['crack_overlap']:.4f}"
    )

    print(
        f"  Total localization score = "
        f"{total_score:.4f}"
    )

    print(
        "\nClassification: "
        f"{classification}"
    )

    # ========================================================
    # SAVE FULL OVERLAY
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / (
            f"localized_crack_deformation_"
            f"{previous_frame}_{target_frame}.png"
        )
    )

    save_overlay(
        points,
        dominant_components,
        target_image,
        crack_mask,
        crack_components,
        output_path,
        (
            previous_frame,
            target_frame,
        ),
        classification,
    )

    print(
        "\nCombined overlay:"
    )

    print(
        output_path
    )

    # ========================================================
    # SAVE CRACK-ONLY OVERLAY
    # ========================================================

    crack_output_path = (
        CRACK_OUTPUT_DIR
        / (
            f"crack_detection_"
            f"{previous_frame}_{target_frame}.png"
        )
    )

    save_crack_only_overlay(
        target_image,
        crack_mask,
        crack_components,
        crack_output_path,
        (
            previous_frame,
            target_frame,
        ),
    )

    print(
        "Crack-only overlay:"
    )

    print(
        crack_output_path
    )

    return {
        "previous_frame":
            previous_frame,
        "target_frame":
            target_frame,
        "classification":
            classification,
        "localization_score":
            total_score,
        "crack_confidence":
            crack_confidence,
        "crack_dic_points":
            crack_dic_points,
        "crack_overlap":
            crack_related_overlap,
    }


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

    results = []

    for previous_frame, target_frame in (
        test_pairs
    ):

        try:

            result = diagnose_pair(
                previous_frame,
                target_frame,
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                "\nERROR for pair "
                f"{previous_frame} → "
                f"{target_frame}:"
            )

            print(
                error
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    if results:

        summary = pd.DataFrame(
            results
        )

        summary_path = (
            OUTPUT_DIR
            / "crack_localized_deformation_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            summary.to_string(
                index=False
            )
        )

        print(
            "\nSummary saved to:"
        )

        print(
            summary_path
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()