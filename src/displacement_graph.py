import pandas as pd
import matplotlib.pyplot as plt

# Load summary
df = pd.read_csv("../output/displacement_summary.csv")

plt.figure(figsize=(12,6))

plt.plot(
    df["Target"],
    df["Average_Displacement"],
    label="Average",
    linewidth=2
)

plt.plot(
    df["Target"],
    df["Maximum_Displacement"],
    label="Maximum",
    linewidth=1
)

plt.xlabel("Image Number")
plt.ylabel("Displacement (pixels)")
plt.title("Displacement vs Image Number")

plt.grid(True)
plt.legend()

plt.savefig("../output/displacement_graph.png", dpi=300)

plt.show()