import pandas as pd

url = "https://www.espn.com/f1/standings/_/season/2025"

tables = pd.read_html(url)

print("Rastu lenteliu kiekis: ", len(tables))
print(tables[0].head())
print(tables[1].head())
