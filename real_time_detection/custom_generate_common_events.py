from obspy.clients.fdsn import Client
import obspy
import matplotlib.pyplot as plt
import re
import pandas as pd
import glob
import argparse
from datetime import datetime


from datetime import datetime, timedelta


# -------------------- User Inputs --------------------
parser = argparse.ArgumentParser(description="Generate common events from daily detection CSVs.")
parser.add_argument("--start", type=str, required=True, help="Start time in UTC, e.g., '2025-12-10 00:00'")
parser.add_argument("--end", type=str, required=True, help="End time in UTC, e.g., '2025-12-10 12:00'")
args = parser.parse_args()

user_start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S")
user_end   = datetime.strptime(args.end,   "%Y-%m-%dT%H:%M:%S")


# -------------------- Find matching CSVs --------------------
all_files = glob.glob("logs/*_to_*_events.csv")
matched_files = []

pattern = re.compile(r".*_(\d{8}_\d{4})_to_(\d{8}_\d{4})_events\.csv$")

file_start = user_start.strftime("%Y%m%d_%H%M")
file_end = user_end.strftime("%Y%m%d_%H%M")

print(file_start)
print(file_end)

for f in all_files:
    m = pattern.match(f)
    if m:
        file_start = datetime.strptime(m.group(1), "%Y%m%d_%H%M")
        file_end   = datetime.strptime(m.group(2), "%Y%m%d_%H%M")
        # Keep files overlapping user range
        if file_end == user_start and file_start == user_end:
            matched_files.append(f)

if not matched_files:
    raise FileNotFoundError("No event files found overlapping the requested time range")

print(matched_files)
print(f"Found {len(matched_files)} event files")





# Load each into a DataFrame
dfs = [pd.read_csv(f) for f in matched_files]

# Check how many files loaded
print(f"Loaded {len(dfs)} event files")


# Combine all station events into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)
print(df_all.head())


# Convert start_time to datetime if it's not already
df_all["start_time"] = pd.to_datetime(df_all["start_time"])



# Round to nearest 10s window (you can try 5s or 15s too)
df_all["rounded_start"] = df_all["start_time"].dt.round("10s")





# Group by the rounded start time
grouped = df_all.groupby("rounded_start").agg(
    num_stations=("station", "nunique"),
    stations=("station", lambda x: list(x)),
    most_common_class=("class", lambda x: x.mode()[0] if not x.mode().empty else "unknown"),
    mean_auc=("auc", "mean"),
    mean_max=("max_prob", "mean"),
    mean_prob=("mean_prob", "mean")
).reset_index()



# Set threshold
N = 4

# Filter the grouped DataFrame
common_events = grouped[grouped["num_stations"] >= N].copy()

# View or save
print(common_events)



# Save output
output_file = f"logs/common_{args.start}_to_{args.end}_events.csv"
common_events.to_csv(output_file, index=False)
print(f"Saved common events to {output_file}")