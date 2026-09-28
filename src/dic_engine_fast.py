import cv2
import numpy as np


def subpixel_offset(left, center, right):
    """
    Parabolic interpolation for sub-pixel accuracy.
    Returns an offset between -0.5 and +0.5 pixels.
    """

    denominator = left - 2.0 * center + right

    if abs(denominator) < 1e-10:
        return 0.0

    return 0.5 * (left - right) / denominator


def track_pair_fast(
    img1,
    img2,
    subset_size=41,
    search_size=81,
    step=10
):
    """
    Faster ZNCC-based DIC tracking.

    The DIC concept remains the same:
        1. Select a subset in img1.
        2. Search for the corresponding subset in img2.
        3. Find the best normalized correlation.
        4. Refine displacement to sub-pixel accuracy.

    OpenCV's optimized template matching is used instead of
    calculating ZNCC manually for every possible position.
    """

    half_subset = subset_size // 2
    half_search = search_size // 2

    margin = half_subset + half_search

    results = []

    # --------------------------------------------------------
    # Keep the same valid tracking region as the old engine
    # --------------------------------------------------------

    for cy in range(
        margin,
        img1.shape[0] - margin,
        step
    ):

        for cx in range(
            margin,
            img1.shape[1] - margin,
            step
        ):

            # ------------------------------------------------
            # Reference subset
            # ------------------------------------------------

            template = img1[
                cy-half_subset:cy+half_subset+1,
                cx-half_subset:cx+half_subset+1
            ]

            if template.shape != (
                subset_size,
                subset_size
            ):
                continue

            # Ignore areas with insufficient texture
            if np.var(template) < 500:
                continue

            # ------------------------------------------------
            # Search region in img2
            #
            # This is the same search concept as the original
            # implementation.
            # ------------------------------------------------

            search_top = cy - half_search
            search_bottom = cy + half_search + 1

            search_left = cx - half_search
            search_right = cx + half_search + 1

            search_region = img2[
                search_top:search_bottom,
                search_left:search_right
            ]

            expected_shape = (
                search_size,
                search_size
            )

            if search_region.shape != expected_shape:
                continue

            # ------------------------------------------------
            # Optimized normalized cross correlation
            # ------------------------------------------------

            correlation = cv2.matchTemplate(
                search_region,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_score, _, max_location = cv2.minMaxLoc(
                correlation
            )

            if np.isnan(max_score):
                continue

            # ------------------------------------------------
            # Same quality threshold as old engine
            # ------------------------------------------------

            if max_score < 0.98:
                continue

            # ------------------------------------------------
            # Calculate integer displacement
            #
            # match location is relative to search_region.
            # Because the search region starts at
            # cx-half_search / cy-half_search, convert it
            # back to image coordinates.
            # ------------------------------------------------

            match_x = search_left + max_location[0]
            match_y = search_top + max_location[1]

            # match location represents the top-left of template
            best_dx = (
                match_x -
                (cx - half_subset)
            )

            best_dy = (
                match_y -
                (cy - half_subset)
            )

            # ------------------------------------------------
            # Same unrealistic displacement rejection
            # as original engine
            # ------------------------------------------------

            if abs(best_dx) > 5 or abs(best_dy) > 5:
                continue

            # ------------------------------------------------
            # Sub-pixel refinement
            # ------------------------------------------------

            center_x = max_location[0]
            center_y = max_location[1]

            center_score = correlation[
                center_y,
                center_x
            ]

            left_score = center_score
            right_score = center_score
            up_score = center_score
            down_score = center_score

            # LEFT
            if center_x > 0:
                left_score = correlation[
                    center_y,
                    center_x - 1
                ]

            # RIGHT
            if center_x < correlation.shape[1] - 1:
                right_score = correlation[
                    center_y,
                    center_x + 1
                ]

            # UP
            if center_y > 0:
                up_score = correlation[
                    center_y - 1,
                    center_x
                ]

            # DOWN
            if center_y < correlation.shape[0] - 1:
                down_score = correlation[
                    center_y + 1,
                    center_x
                ]

            sub_dx = subpixel_offset(
                left_score,
                center_score,
                right_score
            )

            sub_dy = subpixel_offset(
                up_score,
                center_score,
                down_score
            )

            best_dx += sub_dx
            best_dy += sub_dy

            results.append([
                cx,
                cy,
                best_dx,
                best_dy,
                float(max_score)
            ])

    return results