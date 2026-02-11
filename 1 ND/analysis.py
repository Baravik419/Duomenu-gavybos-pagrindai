import pandas as pd

df = pd.read_csv("f1_standings_history.csv")

race_columns = df.columns[4:-1]

df["avg_points_per_race"] = df[race_columns].mean(axis=1)

print("\nLenktyninko vidurkis:")
print(df[["driver_name", "avg_points_per_race"]].head())

print("\nLenktynių vidurkis:")
race_average = df[race_columns].mean()
print(race_average.sort_values(ascending=False))