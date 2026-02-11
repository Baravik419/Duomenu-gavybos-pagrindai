import pandas as pd

url = "https://www.espn.com/f1/standings/_/season/2025"

def read_and_modify():
    tables = pd.read_html(url)

    ds0 = tables[0].copy() # driver standings
    dr1 = tables[1].copy() # driver race results

    ds0 = ds0.rename(columns={"Unnamed: 0": "driver_raw"}) # apkeicia unnamed i driver_raw
    dr1 = dr1.loc[:, ~dr1.columns.astype(str).str.startswith("Unnamed")] #suranda Unnamed, ~ invertuoja is True i False

    #print("ds0 columns:", list(ds0.columns))
    #print("dr1 columns:", list(dr1.columns))

    df = pd.concat([ds0, dr1], axis=1) #concat yra apjungimas, norint jungti per eilutes axis=0

    #print("Formatas:", df.shape)
    #print(df.head())

    df[["driver_position", "driver_code", "driver_name"]] = df["driver_raw"].str.extract(r"(\d+)([A-Z]{3})(.*)") # r"" raw string; d yra skaičius, + keli; [A-Z] visos raides, {3} konkrciai tris kartus;.* wildcard'as; () kiekvienas atskiras stulpelis

    #print(df[["driver_position", "driver_code", "driver_name"]].head())
    #print("df columns:", list(df.columns))

    #gilesnis duomenu apdorojimas
    df.drop(columns = ["driver_raw"]) #istrinam jau panaudota stulpeli
    race_columns = list(dr1.columns)
    df[race_columns] = df[race_columns].replace("-", pd.NA) #- yra string, turim duot suprasti pandai, kad ten nera skaiciaus, pd.NA ta padaro
    df["driver_position"] = pd.to_numeric(df["driver_position"], errors="coerce") #konvertuojam i skaitines reiksmes del duomenu stabilumo
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce") #taip pat konvertuojam del stabilumo

    for column in race_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce") #ir dar karta konvertuojam, bet jau pacius stulpelius

    df = df[["driver_position", "driver_code", "driver_name", "PTS"] + race_columns] #grazinam duomenu rikiavima kaip buvo paciame HTML'e

    return df