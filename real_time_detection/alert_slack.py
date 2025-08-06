#!/usr/bin/env python3
import os, requests, pandas as pd
from datetime import datetime
from pathlib import Path

today = datetime.utcnow().strftime("%Y-%m-%d")


csv = f"/home/ak287/PNW_Seismic_Event_Classification/real_time_detection/logs/common_{today}_events.csv"

try:
    df = pd.read_csv(csv)
    n_surface = (df["most_common_class"] == "su").sum()
    n_earthquake = (df["most_common_class"] == "eq").sum()
    n_explosion = (df["most_common_class"] == "px").sum()
except FileNotFoundError:
    n_surface = "N/A"
    n_earthquake = "N/A"
    n_explosion = "N/A"

msg = (
    f"*{n_surface} surface event(s)* detected at Mt. Rainier on *{today}*.\n"
    f"*{n_earthquake} earthquake event(s)* detected at Mt. Rainier on *{today}*.\n"
    f"*{n_explosion} explosion event(s)* detected at Mt. Rainier on *{today}*.\n"
    f"➡️  <{os.environ['SHEET_URL']}|Click here to review & label>"
)

resp = requests.post(os.environ["SLACK_WEBHOOK"], json={"text": msg})
resp.raise_for_status()
