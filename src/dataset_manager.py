from pathlib import Path
import re
import cv2


# ============================================================
# Project paths
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "BGSL"


# ============================================================
# Frame number extraction
# ============================================================

def get_frame_number(file_path):
    """
    Extract the actual frame number from filenames such as:

        BGSL-2-0001_0.tif  -> 1
        BGSL-2-2764_0.tif  -> 2764
        BGSL-2-6600_0.tif  -> 6600

    The final '_0' is NOT treated as the frame number.
    """

    filename = Path(file_path).name

    match = re.search(r"BGSL-2-(\d+)_0\.tif$", filename, re.IGNORECASE)

    if match is None:
        raise ValueError(
            f"Could not determine frame number from filename: {filename}"
        )

    return int(match.group(1))


# ============================================================
# Find all TIFF images
# ============================================================

def get_image_files():
    """
    Find all TIFF images and sort them by their actual
    experimental frame number.
    """

    files = list(DATASET_DIR.glob("*.tif"))

    frame_files = []

    for file in files:

        try:
            frame_number = get_frame_number(file)
            frame_files.append((frame_number, file))

        except ValueError:
            print(f"Warning: ignoring unexpected filename: {file.name}")

    frame_files.sort(key=lambda item: item[0])

    return frame_files


# ============================================================
# Print dataset information
# ============================================================

def print_dataset_info():

    frame_files = get_image_files()

    print("\n==========================================")
    print("       DATASET INFORMATION")
    print("==========================================")

    print("Dataset directory:")
    print(DATASET_DIR)

    print("\nTotal TIFF images found:", len(frame_files))

    if len(frame_files) == 0:
        print("\nNo TIFF images found.")
        return

    print("\nAvailable frames:")

    for frame_number, file in frame_files[:10]:
        print(f"Frame {frame_number:04d} -> {file.name}")

    if len(frame_files) > 10:
        print("...")
        
        for frame_number, file in frame_files[-5:]:
            print(f"Frame {frame_number:04d} -> {file.name}")

    print("==========================================\n")


# ============================================================
# Get image for a specific frame
# ============================================================

def get_image(frame_number):

    frame_files = get_image_files()

    for number, file in frame_files:

        if number == frame_number:

            image = cv2.imread(
                str(file),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise RuntimeError(
                    f"Could not read image: {file}"
                )

            return image

    raise FileNotFoundError(
        f"Frame {frame_number} was not found in the dataset."
    )


# ============================================================
# Get filename for a specific frame
# ============================================================

def get_image_path(frame_number):

    frame_files = get_image_files()

    for number, file in frame_files:

        if number == frame_number:
            return file

    raise FileNotFoundError(
        f"Frame {frame_number} was not found in the dataset."
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print_dataset_info()

    # Test Frame 1
    try:

        image = get_image(1)

        print(
            f"Frame 1 loaded successfully: "
            f"{image.shape[1]} x {image.shape[0]}"
        )

    except Exception as e:

        print("Frame 1 test failed:")
        print(e)

    # Test Frame 6600
    try:

        image = get_image(6600)

        print(
            f"Frame 6600 loaded successfully: "
            f"{image.shape[1]} x {image.shape[0]}"
        )

    except Exception as e:

        print("Frame 6600 test:")
        print(e)