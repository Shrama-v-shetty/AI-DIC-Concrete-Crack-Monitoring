import pandas as pd
import matplotlib.pyplot as plt

# Load displacement data
df = pd.read_csv("../output/displacement.csv")

plt.figure(figsize=(12,6))

plt.quiver(
    df["X"],
    df["Y"],
    df["dx"],
    -df["dy"],          # invert Y for image coordinates
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red"
)

plt.gca().invert_yaxis()

plt.title("DIC Displacement Vector Field")

plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")

plt.savefig("../output/displacement_vectors.png", dpi=300)

plt.show()

print("Saved displacement_vectors.png")