import pandas as pd
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset"

EXCEL_FILE = r"C:\Users\shrama\Desktop\Dam updated dataset\experimental_data\combined_image_load_data_0_to_6600.xlsx"

DIC_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage3_DIC_Test"

OUTPUT_FOLDER = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset\Stage5_Correlation"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# DIC INTERVALS ALREADY TESTED
# ============================================================

DIC_INTERVALS = [
    (1000, 1274),
    (1274, 2000),
    (6500, 6600)
]


# ============================================================
# FIND THE EXCEL FILE
# ============================================================

if not os.path.exists(EXCEL_FILE):

    print("Excel file not found at:")
    print(EXCEL_FILE)

    print("\nPlease check the filename.")

    raise SystemExit


# ============================================================
# READ EXCEL
# ============================================================

print("==========================================")
print("     STAGE 5 - DIC + EXCEL CORRELATION")
print("==========================================")

print("\nReading Excel:")
print(EXCEL_FILE)

excel = pd.read_excel(EXCEL_FILE)

print("\nRows:", len(excel))

print("\nColumns:")
for column in excel.columns:
    print(" -", column)


# ============================================================
# REQUIRED COLUMN
# ============================================================

if "Image Number" not in excel.columns:

    print("\nERROR: 'Image Number' column not found.")

    raise SystemExit


# ============================================================
# CLEAN IMAGE NUMBER
# ============================================================

excel["Image Number"] = pd.to_numeric(
    excel["Image Number"],
    errors="coerce"
)

excel = excel.dropna(
    subset=["Image Number"]
)

excel["Image Number"] = (
    excel["Image Number"]
    .astype(int)
)


# ============================================================
# FUNCTION TO GET EXCEL DATA FOR A FRAME
# ============================================================

def get_frame_data(frame_number):

    rows = excel[
        excel["Image Number"] == frame_number
    ]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


# ============================================================
# FUNCTION TO READ DIC CSV
# ============================================================

def read_dic(frame1, frame2):

    filename = (
        f"dic_{frame1:04d}_{frame2:04d}.csv"
    )

    filepath = os.path.join(
        DIC_FOLDER,
        filename
    )

    if not os.path.exists(filepath):

        print(
            "\nDIC file not found:",
            filepath
        )

        return None

    df = pd.read_csv(filepath)

    if "Displacement" not in df.columns:

        df["Displacement"] = np.sqrt(
            df["dx"]**2 +
            df["dy"]**2
        )

    result = {

        "DIC_Tracked_Points":
            len(df),

        "DIC_Average_dx":
            df["dx"].mean(),

        "DIC_Average_dy":
            df["dy"].mean(),

        "DIC_Average_Displacement":
            df["Displacement"].mean(),

        "DIC_Max_Displacement":
            df["Displacement"].max(),

        "DIC_Average_ZNCC":
            df["ZNCC"].mean(),

        "DIC_Min_ZNCC":
            df["ZNCC"].min(),

        "DIC_Max_ZNCC":
            df["ZNCC"].max()
    }

    return result


# ============================================================
# PROCESS EACH DIC INTERVAL
# ============================================================

results = []


for frame1, frame2 in DIC_INTERVALS:

    print("\n==========================================")
    print(
        f"Processing interval: "
        f"{frame1} -> {frame2}"
    )
    print("==========================================")


    # --------------------------------------------------------
    # Excel information
    # --------------------------------------------------------

    data1 = get_frame_data(frame1)
    data2 = get_frame_data(frame2)

    if data1 is None:

        print(
            f"Excel data not found for frame {frame1}"
        )

        continue

    if data2 is None:

        print(
            f"Excel data not found for frame {frame2}"
        )

        continue


    # --------------------------------------------------------
    # DIC information
    # --------------------------------------------------------

    dic = read_dic(
        frame1,
        frame2
    )

    if dic is None:
        continue


    # --------------------------------------------------------
    # BASIC EXPERIMENTAL VALUES
    # --------------------------------------------------------

    row = {

        "Frame_Start":
            frame1,

        "Frame_End":
            frame2,

        "Time_Start":
            data1["Time_0"],

        "Time_End":
            data2["Time_0"],

        "Time_Change":
            data2["Time_0"] -
            data1["Time_0"],

        "Load_Start_kN":
            data1["Load ch 2 kN"],

        "Load_End_kN":
            data2["Load ch 2 kN"],

        "Load_Change_kN":
            data2["Load ch 2 kN"] -
            data1["Load ch 2 kN"]
    }


    # --------------------------------------------------------
    # ALL DISPLACEMENT CHANNELS
    # --------------------------------------------------------

    displacement_columns = [

        "Displacement ch 5 µm",
        "Displacement ch 6 µm",
        "Displacement ch 7 µm",
        "Displacement ch 8 µm"
    ]


    for column in displacement_columns:

        if column in excel.columns:

            start_value = pd.to_numeric(
                data1[column],
                errors="coerce"
            )

            end_value = pd.to_numeric(
                data2[column],
                errors="coerce"
            )

            row[column + " Start"] = start_value

            row[column + " End"] = end_value

            row[column + " Change"] = (
                end_value -
                start_value
            )


    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    if "Stress MPa" in excel.columns:

        stress_start = pd.to_numeric(
            data1["Stress MPa"],
            errors="coerce"
        )

        stress_end = pd.to_numeric(
            data2["Stress MPa"],
            errors="coerce"
        )

        row["Stress_Start_MPa"] = stress_start

        row["Stress_End_MPa"] = stress_end

        row["Stress_Change_MPa"] = (
            stress_end -
            stress_start
        )


    # --------------------------------------------------------
    # DIC VALUES
    # --------------------------------------------------------

    row.update(dic)


    results.append(row)


# ============================================================
# CREATE RESULT TABLE
# ============================================================

if len(results) == 0:

    print("\nNo correlation results generated.")

    raise SystemExit


result_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

output_csv = os.path.join(
    OUTPUT_FOLDER,
    "dic_excel_correlation.csv"
)

result_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# SAVE EXCEL
# ============================================================

output_excel = os.path.join(
    OUTPUT_FOLDER,
    "dic_excel_correlation.xlsx"
)

result_df.to_excel(
    output_excel,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n==========================================")
print("          CORRELATION SUMMARY")
print("==========================================")

display_columns = [

    "Frame_Start",
    "Frame_End",
    "Time_Start",
    "Time_End",
    "Load_Start_kN",
    "Load_End_kN",
    "Load_Change_kN",
    "DIC_Average_Displacement",
    "DIC_Max_Displacement",
    "DIC_Average_ZNCC"
]


available_display_columns = [

    column
    for column in display_columns
    if column in result_df.columns
]


print(
    result_df[
        available_display_columns
    ].to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("        STAGE 5 COMPLETE")
print("==========================================")

print("\nCSV saved:")
print(output_csv)

print("\nExcel saved:")
print(output_excel)