import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)

df = pd.read_csv("../1 ND/f1_standings_history.csv")

# Atsifiltruoju stulpelius - bendri taskai ir kiekvienos lenktynes
race_columns = df.columns[3:-1]

# Isvedu TOP 10 pagal surinktus taskus
print("\n TOP 10 pagal taškus:")
sorted_df = df.sort_values(by=["PTS"], ascending=False)
print(sorted_df[["driver_name", "PTS"]].head(10).to_string(index=False, header=False, justify="left"))

print("\n Aprašomoji statistika:")
print(df[race_columns].describe().round(1))

# Matuojam deziu plocius

df["PTS"].plot(kind="box")
plt.title("Taškų pasiskirstymas tarp vairuotojų")
plt.ylabel("Taškai")
plt.show()

# Mokomes istorijos

df["PTS"].hist(bins=10, edgecolor="black")
plt.title("Histograma – pasiskirstymas pagal taškus")
plt.ylabel("Vairuotojų skaičius")
plt.xlabel("Taškų skaičius")
plt.show()

# Testu testas

aus_points = df["AUS"]
aus_points = aus_points.dropna()
chi_points = df["CHN"]
chi_points = chi_points.dropna()

t_stat, p_value = ttest_ind(aus_points, chi_points)

print("\nPirmas T-test'as")
print("T-test rezultatas = ", t_stat)
print("P-value rezultatas = ", p_value)

jpn_points = df["JPN"]
jpn_points = jpn_points.dropna()

t_stat, p_value = ttest_ind(aus_points, jpn_points)

print("\nAntras T-test'as")
print("T-test rezultatas = ", t_stat)
print("P-value rezultatas = ", p_value)

# Korozijos apskaiciavimas

ses_points = df["PTS"]
correlation = ses_points.corr(aus_points)

print("\n Koreliacijos koeficientas: ", correlation)