import json
import os

roi = {
    "x": 0,
    "y": 607,
    "width": 2268,
    "height": 692
}

os.makedirs("../output", exist_ok=True)

with open("../output/beam_roi.json", "w") as f:
    json.dump(roi, f, indent=4)

print("ROI saved successfully.")