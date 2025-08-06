import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime

# ── load today's CSV ────────────────────────────────────────
today = datetime.utcnow().strftime("%Y-%m-%d")
csv_path = f"logs/common_{today}_events.csv"
df = pd.read_csv(csv_path)

# ── 2.  Append analyst-friendly columns (blank) ──────────────────
df["Analyst label"] = ""        # analysts choose: Correct / Incorrect / etc.
# ── 2.  Append analyst-friendly columns (blank) ──────────────────
df["Analyst name"] = ""        # analysts choose: Correct / Incorrect / etc.

df["Comments"]      = ""        # free-text notes




# ── authenticate with service-account key ──────────────────
creds = Credentials.from_service_account_file(
    "service_account.json",
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
client = gspread.authorize(creds)

# ── open the sheet YOU created (no quota hit) ───────────────
SPREADSHEET_ID = "1SJsozRLMOW8QBTNDdANEnrQH33GHfWqa7ahbLPMHfU8"        # <— paste yours
ss = client.open_by_key(SPREADSHEET_ID)

# ── create/replace a tab for today ──────────────────────────
TAB = today
try:                       # delete if it already exists
    ss.del_worksheet(ss.worksheet(TAB))
except gspread.exceptions.WorksheetNotFound:
    pass
ws = ss.add_worksheet(title=TAB, rows=2000, cols=20)

# ── write the DataFrame ─────────────────────────────────────
set_with_dataframe(ws, df)
print(f"✅  Uploaded {csv_path} → tab “{TAB}” in Google Sheet.")
