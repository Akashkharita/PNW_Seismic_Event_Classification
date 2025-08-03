# daily_detection.py

import os
import json
import csv
import numpy as np
import torch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import pandas as pd

from utils.detect import smooth_moving_avg, detect_event_windows


import sys
sys.path.append('/home/ak287/seisbench/seisbench/models')
import seisbench.models as sbm



# -------------------- Setup --------------------

# Load model
model = sbm.QuakeXNet.from_pretrained("base", version_str = '3')


# Load station list
with open("stations.json", "r") as f:
    stations = json.load(f)

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_file = "logs/detections.csv"

# Write header to CSV if not exists
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "station", "final_label", "eq_auc", "px_auc", "su_auc"])

# Time window: yesterday
et_time = UTCDateTime.utcnow()
st_time = et_time - 86400  # 24 hours ago




# -------------------- Main Loop --------------------

client = Client("IRIS")
chn_prefix = "QuakeXNet_"
class_names = ['eq', 'px', 'su']
channel_map = {cls: f"{chn_prefix}{cls}" for cls in class_names}

for entry in stations:
    net = entry["net"]
    sta = entry["sta"]
    chn = entry.get("chn", "*H")

    print(f"🔍 Processing {net}.{sta}...")

    try:
        st = client.get_waveforms(
            network=net,
            station=sta,
            channel=chn + "*",
            location="*",
            starttime=st_time,
            endtime=et_time,
        )

        # Run inference
        probs_st = model.annotate(st, stride=500)

        total_auc = {}
        mean_probs = {}
        max_probs = {}
        start = {}
        end = {}

        for cls in class_names:
            total_auc[cls] = []
            mean_probs[cls] = []
            max_probs[cls] = []
            start[cls] = []
            end[cls] = []

            probs = probs_st.select(channel=channel_map[cls])

            for prob in probs:
                probs_array = np.array(prob)
                s_cls = smooth_moving_avg(probs_array)
                events = detect_event_windows(s_cls)

                for event in events:
                    total_auc[cls].append(event['area_under_curve'])
                    mean_probs[cls].append(event['mean_prob'])
                    max_probs[cls].append(event['max_prob'])
                    start[cls].append(event['start'])
                    end[cls].append(event['end'])
            
            
        
        event_records = []

        for cls in class_names:
            for i in range(len(total_auc[cls])):
                start_idx = start[cls][i]
                end_idx = end[cls][i]
                event_start_time = st_time + (start_idx * 500 / 50)  # stride=500, sampling=50Hz
                event_end_time = st_time + (end_idx * 500 / 50)

                event_records.append({
                    "station": sta,
                    "network": net,
                    "class": cls,
                    "auc": total_auc[cls][i],
                    "mean_prob": mean_probs[cls][i],
                    "max_prob": max_probs[cls][i],
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "start_time": str(event_start_time),
                    "end_time": str(event_end_time),
                })

        # Convert to DataFrame
        if event_records:
            df_events = pd.DataFrame(event_records)
            print(df_events)

            # Optionally save it to CSV
            df_events.to_csv(f"logs/{sta}_{et_time.date}_events.csv", index=False)



    except Exception as e:
        print(f"❌ Error processing {net}.{sta}: {e}")