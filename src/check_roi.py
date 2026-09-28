import cv2
import glob
import os

roi_path = r"C:\Users\shrama\Desktop\Dam updated dataset\dataset\BGSL_ROI"

image_files = sorted(
    glob.glob(os.path.join(roi_path, "*.tif"))
)

print("Total ROI images:", len(image_files))

# Select 3 frames
indices = [
    0,
    len(image_files) // 2,
    len(image_files) - 1
]

for index in indices:

    file = image_files[index]

    img = cv2.imread(file)

    if img is None:
        print("Could not read:", file)
        continue

    print("\nDisplaying:", os.path.basename(file))
    print("Size:", img.shape[1], "x", img.shape[0])

    cv2.imshow(
        f"ROI Frame {index}",
        img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nROI verification complete.")