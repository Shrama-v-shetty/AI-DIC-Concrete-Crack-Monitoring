import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_reference_displacement_graph(input_csv, output_image):

    df = pd.read_csv(input_csv)

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

    plt.savefig(output_image, dpi=300)
    plt.close()

    print("Graph Saved :", output_image)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    generate_reference_displacement_graph(
        os.path.join(OUTPUT_DIR, "reference_displacement_summary.csv"),
        os.path.join(OUTPUT_DIR, "reference_displacement_graph.png")
    )