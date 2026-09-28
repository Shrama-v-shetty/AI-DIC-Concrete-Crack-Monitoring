import cv2
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

IMAGE_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\BGSL_ROI"
)

EVENT_CSV = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6_Event_Analysis"
    r"\candidate_events.csv"
)

OUTPUT_FOLDER = (
    r"C:\Users\shrama\Desktop\Dam updated dataset"
    r"\Dataset\Stage6_Event_Analysis"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Number of frames before and after an event to display
OFFSET = 20

# Number of automatically selected events to preview
NUMBER_OF_EVENTS = 10

# Important manually selected mechanical event
# We already observed interesting load behaviour around this frame.
MANUAL_EVENTS = [1700]


# ============================================================
# FUNCTION: GET ACTUAL FRAME NUMBER
# ============================================================

def get_frame_number(filepath):

    filename = os.path.basename(filepath)

    numbers = re.findall(r"\d+", filename)

    if len(numbers) < 2:
        return None

    # IMPORTANT:
    # BGSL-2-1274_0.tif
    #
    # numbers = ['2', '1274', '0']
    #
    # Actual image/frame number = second-last number

    return int(numbers[-2])


# ============================================================
# START
# ============================================================

print("==========================================")
print("      STAGE 6B - EVENT PREVIEW")
print("==========================================")


# ============================================================
# LOAD ALL AVAILABLE ROI IMAGES
# ============================================================

all_files = glob.glob(
    os.path.join(IMAGE_FOLDER, "*.tif")
)

frame_files = {}

for file in all_files:

    frame = get_frame_number(file)

    if frame is None:
        continue

    # Ignore frame 0
    if frame == 0:
        continue

    frame_files[frame] = file


available_frames = sorted(frame_files.keys())


print("\nAvailable ROI frames:", len(available_frames))

if len(available_frames) > 0:

    print("First available frame:", available_frames[0])
    print("Last available frame :", available_frames[-1])


# ============================================================
# LOAD EVENT CANDIDATES
# ============================================================

if not os.path.exists(EVENT_CSV):

    print("\nERROR:")
    print("Candidate event file was not found:")
    print(EVENT_CSV)

    print("\nRun Stage 6 first.")

    raise SystemExit


events_df = pd.read_csv(EVENT_CSV)


print("\nTotal Excel event candidates:", len(events_df))


# ============================================================
# CHECK WHICH EVENTS HAVE IMAGES
# ============================================================

image_verified_events = []

missing_image_events = []


for _, row in events_df.iterrows():

    frame = int(row["Image Number"])

    if frame in frame_files:

        image_verified_events.append(row)

    else:

        missing_image_events.append(row)


print(
    "Events with available images:",
    len(image_verified_events)
)

print(
    "Events inside missing image ranges:",
    len(missing_image_events)
)


# ============================================================
# SELECT TOP EVENTS THAT HAVE ACTUAL IMAGES
# ============================================================

image_verified_df = pd.DataFrame(
    image_verified_events
)


if len(image_verified_df) > 0:

    # Sort according to Stage 6 event score
    image_verified_df = image_verified_df.sort_values(
        by="Event_Score",
        ascending=False
    )

    automatic_events = (
        image_verified_df
        .head(NUMBER_OF_EVENTS)
        ["Image Number"]
        .astype(int)
        .tolist()
    )

else:

    automatic_events = []


# ============================================================
# ADD MANUAL IMPORTANT EVENTS
# ============================================================

event_frames = []

for frame in MANUAL_EVENTS:

    if frame in frame_files:

        event_frames.append(frame)

    else:

        print(
            f"\nManual event {frame} has no available image."
        )


for frame in automatic_events:

    if frame not in event_frames:

        event_frames.append(frame)


# Sort events
event_frames = sorted(event_frames)


# ============================================================
# DISPLAY SELECTED EVENTS
# ============================================================

print("\n==========================================")
print("IMAGE-VERIFIABLE EVENTS")
print("==========================================")

for frame in event_frames:

    if frame in MANUAL_EVENTS:

        print(
            f"Frame {frame}  -> MANUAL MECHANICAL EVENT"
        )

    else:

        matching = image_verified_df[
            image_verified_df["Image Number"] == frame
        ]

        if len(matching) > 0:

            score = matching.iloc[0]["Event_Score"]

            print(
                f"Frame {frame}  -> Event Score: {score:.3f}"
            )


# ============================================================
# SAVE IMAGE-VERIFIED EVENT LIST
# ============================================================

verified_output = os.path.join(
    OUTPUT_FOLDER,
    "image_verified_events.csv"
)

if len(image_verified_df) > 0:

    image_verified_df.to_csv(
        verified_output,
        index=False
    )

    print(
        "\nSaved image-verified event list:"
    )

    print(verified_output)


# ============================================================
# GENERATE EVENT PREVIEWS
# ============================================================

print("\n==========================================")
print("GENERATING EVENT PREVIEWS")
print("==========================================")


for event_frame in event_frames:

    # --------------------------------------------------------
    # Frames before / event / after
    # --------------------------------------------------------

    frames_to_show = [
        event_frame - OFFSET,
        event_frame,
        event_frame + OFFSET
    ]

    available = []

    for frame in frames_to_show:

        if frame in frame_files:

            available.append(frame)


    if len(available) == 0:

        print(
            f"\nNo images available around frame {event_frame}"
        )

        continue


    print("\n==========================================")
    print("Event frame:", event_frame)
    print("Available preview frames:", available)
    print("==========================================")


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(15, 5)
    )


    if len(available) == 1:

        axes = [axes]


    # --------------------------------------------------------
    # Plot images
    # --------------------------------------------------------

    for ax, frame in zip(axes, available):

        filepath = frame_files[frame]

        img = cv2.imread(
            filepath,
            cv2.IMREAD_GRAYSCALE
        )


        if img is None:

            print(
                "Could not read:",
                filepath
            )

            continue


        ax.imshow(
            img,
            cmap="gray"
        )


        if frame == event_frame:

            ax.set_title(
                f"EVENT FRAME {frame}",
                fontsize=12
            )

        else:

            ax.set_title(
                f"Frame {frame}",
                fontsize=11
            )


        ax.axis("off")


    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    if event_frame in MANUAL_EVENTS:

        title = (
            f"Important Mechanical Event - Frame {event_frame}"
        )

    else:

        title = (
            f"Candidate Event Around Frame {event_frame}"
        )


    fig.suptitle(
        title,
        fontsize=14
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # Save preview
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"event_preview_{event_frame}.png"
    )


    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        output_file
    )


# ============================================================
# PRINT MISSING EVENT INFORMATION
# ============================================================

print("\n==========================================")
print("EVENTS WITHOUT IMAGES")
print("==========================================")


if len(missing_image_events) == 0:

    print("None")


else:

    # Show first 15 missing-image candidates
    for row in missing_image_events[:15]:

        print(
            f"Frame {int(row['Image Number'])}"
        )


    if len(missing_image_events) > 15:

        print(
            f"... and "
            f"{len(missing_image_events) - 15} more"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("       STAGE 6B COMPLETE")
print("==========================================")

print(
    "\nTotal available image frames:",
    len(available_frames)
)

print(
    "Total Excel event candidates:",
    len(events_df)
)

print(
    "Image-verifiable candidates:",
    len(image_verified_df)
)

print(
    "Selected events for preview:",
    len(event_frames)
)

print("\nPreview files are saved at:")

print(OUTPUT_FOLDER)

print("\nNext step:")
print(
    "Inspect the generated event_preview_*.png files."
)

print(
    "These will be used to decide which events should "
    "be validated using DIC."
)