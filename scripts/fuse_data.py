import pandas as pd
import time

# --------------------- CONFIGURATION ---------------------
EXPORT_DATA = True
REMOVE_OUTLIERS = True
VIEW_INFO = False #For debugging

OUTPUT_PATH = "../source_files/"
OUTPUT_NAME = "combined_data"

EXCLUDE_FROM_OUTLIERS = ["zone", "on_1b", "on_2b", "on_3b", "top"]
PITCH_AND_CODE_MIN = 1000

#Fielders choice, fielders choice out, intent walk, Field Error, Sac Fly, Sac Bunt batter interference?

EVENT_TO_DROP = ["Field Error", "Fielders Choice Out", "Intent Walk", "Fielders Choice" \
    "Bunt Pop Out", "Catcher Interference", "Batter Interference", "Bunt Lineout", \
    "Sacrifice Bunt DP", "Fielders Choice"]

#-1 Off
#0 Remove only meaningless columns
#1 For pitch/batting decision analysis
#2 For pitch identification
COLUMN_REMOVAL_SWITCH = 1

# --------------------- CONFIGURATION PRE-PROCESSING ---------------------
COL_TO_DROP = ["type_confidence", "y0", "nasty", "event_num", "ab_id", "batter_id", \
        "g_id", "o", "pitcher_id"]

match COLUMN_REMOVAL_SWITCH:
    case -1:
        COL_TO_DROP = []
    case 1:
        COL_TO_DROP = COL_TO_DROP + ["end_speed", "break_angle", "break_length",
            "break_y", "ax", "ay", "az", "sz_bot", "sz_top", "vx0", "vy0",
            "vz0", "x", "x0", "y", "z0", "pfx_x", "pfx_z", "type"]
    case 2:
        pass
    case _:
        raise ValueError("Invalid COLUMN_REMOVAL_SWITCH value")

OUTPUT_NAME = OUTPUT_NAME + "_C" + str(COLUMN_REMOVAL_SWITCH) + ".hdf5"

# --------------------- DATA IMPORT ---------------------
start_time = time.time()
print("Loading source files")
atBats = pd.read_csv("../source_files/atbats.csv")
pitches = pd.read_csv("../source_files/pitches.csv")

elapsed = time.time() - start_time
print(f"Source Files Loaded, Elapsed time: {elapsed:.3}s")

# --------------------- DATA REDUCTION & VALIDATION ---------------------
print("Beginning data reduction")
combined_data = pitches.merge(atBats, how='inner', on='ab_id')
print(f"Size: {combined_data.shape}")

combined_data = combined_data.drop(
    COL_TO_DROP,
    axis=1)
print(f"Size after dropping columns: {combined_data.shape}")

combined_data = combined_data.dropna()
print(f"Size after NaN removal: {combined_data.shape}")

#Outlier removal
if REMOVE_OUTLIERS:
    for column in combined_data.columns:
        if isinstance((combined_data[column])[0], str):
            continue

        if column in EXCLUDE_FROM_OUTLIERS:
            continue

        upper = combined_data[column].mean() + 3*combined_data[column].std()
        lower = combined_data[column].mean() - 3*combined_data[column].std()

        combined_data = combined_data[(lower < combined_data[column]) & 
            (combined_data[column] < upper)]
        if VIEW_INFO: print(f"Size after {column:>13} removal: {combined_data.shape}")

print(f"Size after outlier removal: {combined_data.shape}")

#Remove Outlier Pitches
pitch_counts = combined_data["pitch_type"].value_counts().rename_axis('unique_values').reset_index(name='counts')
combined_data = combined_data[combined_data["pitch_type"].isin(
    (pitch_counts["unique_values"])[pitch_counts["counts"] > PITCH_AND_CODE_MIN].tolist())
    ]
    
code_counts = combined_data["code"].value_counts().rename_axis('unique_values').reset_index(name='counts')
combined_data = combined_data[combined_data["code"].isin( 
    (code_counts["unique_values"])[code_counts["counts"] > PITCH_AND_CODE_MIN].tolist())
    ]  

print(f"Size after outlier pitch/code removal: {combined_data.shape}")

#REMOVE 
#Fielders choice, fielders choice out, intent walk, Field Error, Sac Fly, Sac Bunt batter interference?
combined_data = combined_data[~combined_data["event"].isin(EVENT_TO_DROP)]

elapsed = time.time() - start_time
print(f"Reduction complete, Elapsed time: {elapsed:.3}s")
print(f"Shape: {combined_data.shape}")

# --------------------- At bat score ---------------------
def atBatScore(event, on_1b, on_2b, on_3b):
    match event:
        case "Groundout" | "Runner out" | "Flyout" | "Forceout" | "Pop out" \
            | "Lineout" | "Grounded Into DP" | "Pop Out" | "Triple Play" \
            | "Bunt Groundout" | "Field Error" | "Double Play" | "Runner Out" \
            | "Bunt Pop Out":
            return 1
        case "Strikeout" | "Strikeout - DP":
            return 2
        case "Walk" | "Hit By Pitch":
            return -(1 + on_1b + (on_2b * on_1b) + (on_3b * on_2b * on_1b))
        case "Single" | "Sac Fly" | "Sac Bunt" | "Sac Fly DP":
            return -(1 + sum([on_1b, on_2b, on_3b]))
        case "Double":
            return -(2 + on_3b + 2 * sum([on_1b, on_2b]))
        case "Triple":
            return -(3 + on_3b + 2*on_2b + 3*on_3b)
        case "Home Run":
            return -(4 + on_3b + 2*on_2b + 3*on_3b)
        case _:
            return event

if COLUMN_REMOVAL_SWITCH == 1:
    combined_data["at_bat_score"] = [atBatScore(event, on_1b, on_2b, on_3b) 
    for event, on_1b, on_2b, on_3b 
    in zip(combined_data["event"], combined_data["on_1b"], combined_data["on_2b"], combined_data["on_3b"])]

print(f"At bat scores calculated, elapsed time: {elapsed:.3}s")
print(f"Shape: {combined_data.shape}")
# --------------------- EXPORT ---------------------
if EXPORT_DATA:
    print("Beginning data export")
    combined_data.to_hdf(OUTPUT_PATH + OUTPUT_NAME, key = "df")

    elapsed = time.time() - start_time
    print(f"Export complete, Elapsed time: {elapsed:.3}s")