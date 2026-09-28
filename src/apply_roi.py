import cv2
import glob
import os

dataset_path = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL"

output_path = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\BGSL_ROI"

os.makedirs(output_path, exist_ok=True)

# ROI selected from the second image
x = 0
y = 669
w = 1750
h = 638

# Get all TIFF images in correct order
image_files = sorted(
    glob.glob(os.path.join(dataset_path, "*.tif"))
)

print("==========================================")
print("       APPLYING ROI TO ALL FRAMES")
print("==========================================")

print("Total images found:", len(image_files))

# Skip 0th image
image_files = image_files[1:]

print("Images to process:", len(image_files))
print("ROI:", x, y, w, h)

for i, file in enumerate(image_files):

    img = cv2.imread(file)

    if img is None:
        print("Could not read:", file)
        continue

    # Crop ROI
    beam = img[y:y+h, x:x+w]

    filename = os.path.basename(file)

    output_file = os.path.join(
        output_path,
        filename
    )

    cv2.imwrite(output_file, beam)

    if i % 100 == 0:
        print(
            f"Processed {i+1}/{len(image_files)} : {filename}"
        )

print("\n==========================================")
print("             COMPLETE")
print("==========================================")

print("ROI images saved at:")
print(output_path)