import numpy as np
from zncc import zncc


def subpixel_offset(left, center, right):
    """
    Parabolic interpolation for sub-pixel accuracy.
    Returns offset between -0.5 and +0.5 pixels.
    """
    denominator = left - 2 * center + right

    if abs(denominator) < 1e-10:
        return 0.0

    return 0.5 * (left - right) / denominator


def track_pair(
    img1,
    img2,
    subset_size=41,
    search_size=81,
    step=10
):

    half_subset = subset_size // 2
    half_search = search_size // 2

    margin = half_subset + half_search

    results = []

    # ==========================================
    # DIC Tracking
    # ==========================================

    for cy in range(margin, img1.shape[0] - margin, step):

        for cx in range(margin, img1.shape[1] - margin, step):

            # Reference subset
            template = img1[
                cy-half_subset:cy+half_subset+1,
                cx-half_subset:cx+half_subset+1
            ]

            if template.shape != (subset_size, subset_size):
                continue

            if np.var(template) < 500:
                continue

            best_score = -1
            best_dx = 0
            best_dy = 0

            # ==========================================
            # Integer Pixel Search
            # ==========================================

            for dy in range(
                -half_search + half_subset,
                half_search - half_subset + 1
            ):

                for dx in range(
                    -half_search + half_subset,
                    half_search - half_subset + 1
                ):

                    yy = cy + dy
                    xx = cx + dx

                    window = img2[
                        yy-half_subset:yy+half_subset+1,
                        xx-half_subset:xx+half_subset+1
                    ]

                    if window.shape != (subset_size, subset_size):
                        continue

                    score = zncc(template, window)

                    if np.isnan(score):
                        continue

                    if score > best_score:
                        best_score = score
                        best_dx = dx
                        best_dy = dy

            # Reject poor matches
            if best_score < 0.98:
                continue

            # Reject unrealistic displacement
            if abs(best_dx) > 5 or abs(best_dy) > 5:
                continue

            # ==========================================
            # Sub-pixel Refinement
            # ==========================================

            center = best_score

            left = center
            right = center
            up = center
            down = center

            # LEFT
            yy = cy + best_dy
            xx = cx + best_dx - 1

            window = img2[
                yy-half_subset:yy+half_subset+1,
                xx-half_subset:xx+half_subset+1
            ]

            if window.shape == (subset_size, subset_size):
                left = zncc(template, window)

            # RIGHT
            xx = cx + best_dx + 1

            window = img2[
                yy-half_subset:yy+half_subset+1,
                xx-half_subset:xx+half_subset+1
            ]

            if window.shape == (subset_size, subset_size):
                right = zncc(template, window)

            # UP
            yy = cy + best_dy - 1
            xx = cx + best_dx

            window = img2[
                yy-half_subset:yy+half_subset+1,
                xx-half_subset:xx+half_subset+1
            ]

            if window.shape == (subset_size, subset_size):
                up = zncc(template, window)

            # DOWN
            yy = cy + best_dy + 1

            window = img2[
                yy-half_subset:yy+half_subset+1,
                xx-half_subset:xx+half_subset+1
            ]

            if window.shape == (subset_size, subset_size):
                down = zncc(template, window)

            sub_dx = subpixel_offset(left, center, right)
            sub_dy = subpixel_offset(up, center, down)

            best_dx += sub_dx
            best_dy += sub_dy

            results.append([
                cx,
                cy,
                best_dx,
                best_dy,
                best_score
            ])

    # ==========================================
    # Debug Information
    # ==========================================

    if len(results) > 0:

        scores = np.array([r[4] for r in results])

        dxs = np.array([r[2] for r in results])
        dys = np.array([r[3] for r in results])

        print("\n========== DIC Statistics ==========")
        print("Tracked Points :", len(results))
        print("Average ZNCC   :", np.mean(scores))
        print("Minimum ZNCC   :", np.min(scores))
        print("Maximum ZNCC   :", np.max(scores))

        print("\nAverage dx :", np.mean(dxs))
        print("Average dy :", np.mean(dys))

        print("\nMaximum dx :", np.max(dxs))
        print("Minimum dx :", np.min(dxs))

        print("Maximum dy :", np.max(dys))
        print("Minimum dy :", np.min(dys))

        print("\nFirst 10 Tracked Points")
        print("--------------------------------")

        for r in results[:10]:
            print(
                f"X={r[0]:4d}  "
                f"Y={r[1]:4d}  "
                f"dx={r[2]:7.3f}  "
                f"dy={r[3]:7.3f}  "
                f"ZNCC={r[4]:.4f}"
            )

        print("====================================\n")

    return results