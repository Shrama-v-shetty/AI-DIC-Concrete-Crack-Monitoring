import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "BGSL_ROI")
DIC_DIR = os.path.join(BASE_DIR, "Stage6C_Crack_Progression")
MASTER_CSV = os.path.join(BASE_DIR, "Stage10_DIC_Excel_Integration", "master_dic_excel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Stage12A_Significant_Region_Validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_1 = 1274
FRAME_2 = 1500

GRID_STEP = 10
GAUSSIAN_SIGMA = 2
FIXTURE_X1, FIXTURE_X2 = 930, 1080
FIXTURE_Y1, FIXTURE_Y2 = 450, 638


def extract_frame_number(filename):
    nums = re.findall(r"\d+", os.path.basename(filename))
    return int(nums[-2]) if len(nums) >= 2 else None


def find_image(frame):
    for name in os.listdir(IMAGE_DIR):
        if name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            if extract_frame_number(name) == frame:
                return os.path.join(IMAGE_DIR, name)
    return None


def find_dic_csv(frame1, frame2):
    """
    Find the accumulated DIC CSV produced during Stage 6C.
    Example:
        frame1=1274, frame2=1500
        -> dic_1274_1500.csv
    """

    filename = f"dic_{frame1}_{frame2}.csv"
    path = os.path.join(DIC_DIR, filename)

    if os.path.exists(path):
        return path

    return None


def calculate_strain(df):
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("x", "y", "dx", "dy"):
            rename[c] = lc.upper() if lc in ("x", "y") else lc
    df = df.rename(columns=rename)

    required = ["X", "Y", "dx", "dy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing DIC columns: {missing}. Found: {list(df.columns)}")

    x = pd.to_numeric(df["X"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
    dx = pd.to_numeric(df["dx"], errors="coerce").to_numpy()
    dy = pd.to_numeric(df["dy"], errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    x, y, dx, dy = x[valid], y[valid], dx[valid], dy[valid]

    if len(x) < 50:
        raise RuntimeError("Too few valid DIC points.")

    gxv = np.arange(np.floor(x.min()), np.ceil(x.max()) + GRID_STEP, GRID_STEP)
    gyv = np.arange(np.floor(y.min()), np.ceil(y.max()) + GRID_STEP, GRID_STEP)
    gx, gy = np.meshgrid(gxv, gyv)

    gdx = griddata((x, y), dx, (gx, gy), method="linear")
    gdy = griddata((x, y), dy, (gx, gy), method="linear")

    for arr, values in ((gdx, dx), (gdy, dy)):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = griddata(
                (x, y), values,
                (gx[bad], gy[bad]),
                method="nearest"
            )

    gdx = gaussian_filter(gdx, sigma=GAUSSIAN_SIGMA)
    gdy = gaussian_filter(gdy, sigma=GAUSSIAN_SIGMA)

    du_dy, du_dx = np.gradient(gdx, GRID_STEP, GRID_STEP)
    dv_dy, dv_dx = np.gradient(gdy, GRID_STEP, GRID_STEP)

    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)

    mean = 0.5 * (exx + eyy)
    radius = np.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2)
    principal = mean + radius

    return gx, gy, principal


def colormap():
    return LinearSegmentedColormap.from_list(
        "dic_heatmap",
        ["purple", "blue", "cyan", "green", "yellow", "orange", "red"],
        N=256
    )


print("=" * 65)
print("STAGE 12A - SIGNIFICANT REGION VALIDATION")
print("=" * 65)
print(f"Pair: {FRAME_1} -> {FRAME_2}")

img1 = find_image(FRAME_1)
img2 = find_image(FRAME_2)
dic_path = find_dic_csv(FRAME_1, FRAME_2)

print("Frame 1 image:", img1)
print("Frame 2 image:", img2)
print("DIC CSV:", dic_path)

if not img1 or not img2 or not dic_path:
    raise FileNotFoundError("Required image/DIC file was not found.")

image2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
if image2 is None:
    raise RuntimeError("Could not read end-frame image.")

dic = pd.read_csv(dic_path)
gx, gy, strain = calculate_strain(dic)

stat_mask = ~(
    (gx >= FIXTURE_X1) & (gx <= FIXTURE_X2) &
    (gy >= FIXTURE_Y1) & (gy <= FIXTURE_Y2)
)

valid = np.isfinite(strain) & stat_mask
values = strain[valid]

mean_s = float(np.mean(values))
p95 = float(np.percentile(values, 95))
p99 = float(np.percentile(values, 99))
max_s = float(np.max(values))

high = valid & (strain >= p99)
hy, hx = np.where(high)

if len(hx):
    cx = float(np.mean(gx[hy, hx]))
    cy = float(np.mean(gy[hy, hx]))
else:
    cx = cy = np.nan

print()
print("STRAIN STATISTICS")
print(f"Mean : {mean_s:.6f}")
print(f"P95  : {p95:.6f}")
print(f"P99  : {p99:.6f}")
print(f"Max  : {max_s:.6f}")

if len(hx):
    print()
    print("HIGH-STRAIN REGION")
    print(f"Cells    : {len(hx)}")
    print(f"Centroid : ({cx:.1f}, {cy:.1f})")
    print(f"X range  : {gx[hy,hx].min():.1f} -> {gx[hy,hx].max():.1f}")
    print(f"Y range  : {gy[hy,hx].min():.1f} -> {gy[hy,hx].max():.1f}")

master = pd.read_csv(MASTER_CSV)
row = master[
    (pd.to_numeric(master["Frame_1"], errors="coerce") == FRAME_1) &
    (pd.to_numeric(master["Frame_2"], errors="coerce") == FRAME_2)
]

print()
print("EXPERIMENTAL DATA")
if not row.empty:
    r = row.iloc[0]
    for c in ["Time_1", "Time_2", "Load_1_kN", "Load_2_kN",
              "Displacement_ch7_1_um", "Displacement_ch7_2_um"]:
        if c in r.index:
            print(f"{c}: {r[c]}")
else:
    print("No matching Stage 10 row found.")

summary = pd.DataFrame([{
    "Frame_1": FRAME_1,
    "Frame_2": FRAME_2,
    "Mean_Principal_Strain": mean_s,
    "P95_Principal_Strain": p95,
    "P99_Principal_Strain": p99,
    "Max_Principal_Strain": max_s,
    "High_Strain_Cells": len(hx),
    "High_Strain_Centroid_X": cx,
    "High_Strain_Centroid_Y": cy
}])
summary_path = os.path.join(OUTPUT_DIR, f"summary_{FRAME_1}_{FRAME_2}.csv")
summary.to_csv(summary_path, index=False)

field = pd.DataFrame({
    "X": gx.ravel(),
    "Y": gy.ravel(),
    "Principal_Strain": strain.ravel()
})
field_path = os.path.join(OUTPUT_DIR, f"strain_field_{FRAME_1}_{FRAME_2}.csv")
field.to_csv(field_path, index=False)

cmap = colormap()
vmax = p99

# Full heatmap
plt.figure(figsize=(12, 7))
plt.imshow(image2, cmap="gray")
plt.imshow(
    strain,
    extent=[gx.min(), gx.max(), gy.max(), gy.min()],
    cmap=cmap, alpha=0.58, vmin=0, vmax=vmax
)
plt.colorbar(label="Principal strain")
plt.gca().add_patch(plt.Rectangle(
    (FIXTURE_X1, FIXTURE_Y1),
    FIXTURE_X2 - FIXTURE_X1,
    FIXTURE_Y2 - FIXTURE_Y1,
    fill=False, linestyle="--", linewidth=2
))
if np.isfinite(cx):
    plt.scatter([cx], [cy], marker="x", s=100, linewidths=2)
plt.title(f"DIC Principal Strain Heatmap: {FRAME_1} -> {FRAME_2}")
plt.xlabel("Image X (pixels)")
plt.ylabel("Image Y (pixels)")
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{FRAME_1}_{FRAME_2}.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()

# Validation figure
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(image2, cmap="gray")
axes[0].set_title(f"Original frame {FRAME_2}")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

im = axes[1].imshow(image2, cmap="gray")
axes[1].imshow(
    strain,
    extent=[gx.min(), gx.max(), gy.max(), gy.min()],
    cmap=cmap, alpha=0.58, vmin=0, vmax=vmax
)
axes[1].add_patch(plt.Rectangle(
    (FIXTURE_X1, FIXTURE_Y1),
    FIXTURE_X2 - FIXTURE_X1,
    FIXTURE_Y2 - FIXTURE_Y1,
    fill=False, linestyle="--", linewidth=2
))
if np.isfinite(cx):
    axes[1].scatter([cx], [cy], marker="x", s=100, linewidths=2)
axes[1].set_title("DIC strain heatmap")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

axes[2].imshow(image2, cmap="gray")
axes[2].imshow(
    strain,
    extent=[gx.min(), gx.max(), gy.max(), gy.min()],
    cmap=cmap, alpha=0.62, vmin=0, vmax=vmax
)
axes[2].set_xlim(max(0, FIXTURE_X1 - 350), min(image2.shape[1], FIXTURE_X2 + 350))
axes[2].set_ylim(min(image2.shape[0], FIXTURE_Y2 + 100), max(0, FIXTURE_Y1 - 400))
axes[2].set_title("Central deformation region")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")

fig.colorbar(im, ax=axes, shrink=0.8, label="Principal strain")
fig.suptitle(f"Stage 12A Validation: {FRAME_1} -> {FRAME_2}")
fig.tight_layout()

validation_path = os.path.join(
    OUTPUT_DIR, f"validation_{FRAME_1}_{FRAME_2}.png"
)
fig.savefig(validation_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print()
print("OUTPUTS")
print(heatmap_path)
print(validation_path)
print(summary_path)
print(field_path)
print()
print("STAGE 12A COMPLETE")
print("Review the validation image before proceeding to the next region.")
