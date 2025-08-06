import pandas as pd
import glob
from obspy.clients.fdsn import Client
import obspy
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

# Compute yesterday’s date in the same format as filenames
yesterday = (datetime.utcnow() - timedelta(days=0)).strftime("%Y-%m-%d")

# Glob files from logs folder using that date
event_files = glob.glob(f"logs/*_{yesterday}_events.csv")


# Load each into a DataFrame
dfs = [pd.read_csv(f) for f in event_files]

# Check how many files loaded
print(f"Loaded {len(dfs)} event files")


# Combine all station events into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)



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



common_events.to_csv(f"logs/common_{str(yesterday)}_events.csv", index=False)