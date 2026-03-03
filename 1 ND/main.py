import read_html_F1
import time
from datetime import datetime
import os

print("Working dir:", os.getcwd())
base_dir = os.path.dirname(__file__)
output_file = os.path.join(base_dir, "f1_standings_history.csv")

runs = 7
delay = 1

for i in range(runs):
    df = read_html_F1.read_and_modify()

    df["scraped_at"] = datetime.now().isoformat(timespec="seconds")

    df.to_csv(output_file, mode="w", index=False)

    print(f"[{i+1}/{runs}] Įrašyta {len(df)} eilučių")

    first_write = False

    if i < runs - 1:
        time.sleep(delay)