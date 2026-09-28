import pandas as pd
import matplotlib.pyplot as plt

def generate_frame_displacement_graph(input_csv, output_image):

    df = pd.read_csv(input_csv)

    plt.figure(figsize=(12,6))

    plt.plot(
        df["Target"],
        df["Cumulative_Average_Displacement"],
        linewidth=2,
        color="blue",
        label="Cumulative Average"
    )

    plt.xlabel("Image Number")
    plt.ylabel("Displacement (pixels)")
    plt.title("Cumulative Beam Displacement")

    plt.grid(True)

    plt.legend()

    plt.savefig(output_image, dpi=300)

    plt.close()

    print("Saved :", output_image)


if __name__ == "__main__":

    generate_frame_displacement_graph(
        "../output/frame_displacement_summary.csv",
        "../output/frame_displacement_graph.png"
    )