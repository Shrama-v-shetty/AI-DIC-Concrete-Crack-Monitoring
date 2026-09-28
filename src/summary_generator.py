import glob
import os
import pandas as pd
import numpy as np

# Folder containing all CSV files
folder = "../output"

files = sorted(glob.glob(os.path.join(folder, "frame_*.csv")))

summary = []

print("Total CSV Files :", len(files))

for file in files:

    df = pd.read_csv(file)

    displacement = np.sqrt(df["dx"]**2 + df["dy"]**2)

    avg_disp = displacement.mean()
    max_disp = displacement.max()

    name = os.path.basename(file)

    # frame_0003_0004.csv
    parts = name.replace(".csv", "").split("_")

    reference = int(parts[1])
    target = int(parts[2])

    summary.append([
        reference,
        target,
        avg_disp,
        max_disp
    ])

summary_df = pd.DataFrame(
    summary,
    columns=[
        "Reference",
        "Target",
        "Average_Displacement",
        "Maximum_Displacement"
    ]
)

summary_df.to_csv(
    "../output/displacement_summary.csv",
    index=False
)

print(summary_df.head())

print("\nSaved : displacement_summary.csv")