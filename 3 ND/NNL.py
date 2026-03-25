from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print(df.describe())
df.to_csv("wine_dataset.csv", index=False)

feature_range = (-1,1)
scaler = MinMaxScaler(feature_range=feature_range)

X = df.drop(columns=["target"])
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["target"] = wine.target

#print(df_scaled)
#print(df_scaled.min())
#print(df_scaled.max())

X = df_scaled.drop(columns=["target"])
y = df_scaled["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=df_scaled["target"])

df_train_split = pd.DataFrame(X_train, columns=X_train.columns)
df_train_split["target"] = y_train
df_test_split = pd.DataFrame(X_test, columns=X_test.columns)
df_test_split["target"] = y_test

#print(df_train_split)
#print(df_test_split)

df_train_split.to_csv("wine_dataset_train_split.csv", index=False)
df_test_split.to_csv("wine_dataset_test_split.csv", index=False)

#nu ka, pradedam mokintis

model = MLPClassifier(hidden_layer_sizes=(3,4), random_state=42, max_iter=1000)

model.fit(X_train, y_train)

#modelio bandymas atspėti rezultatus
y_pred = model.predict(X_test)
print(y_pred)
#modelio informacija
print("Modelio parametrai:")
#per kiek iteraciju modelis apsimoko
print("Kiek iteracijų prireikė modelio apmokymui: ", model.n_iter_)
#kiek neuronu yra output sluoksnyje
print("Kiek neuronų yra išėjimo sluoksnyje: ", model.n_outputs_)
#tikslumo matavimas
accuracy = accuracy_score(y_test, y_pred)
print("Modelio tikslumas: ", accuracy)

w1 = model.coefs_[0]
b1 = model.intercepts_[0]
w3 = model.coefs_[2]
b3 = model.intercepts_[2]

df_w1 = pd.DataFrame(w1, index=X.columns, columns=["N1", "N2", "N3"])
df_bias1 = pd.DataFrame([b1], index=["Threshold"], columns=["N1", "N2", "N3"])
df_w3 = pd.DataFrame(w3, index=["N1", "N2", "N3", "N4"], columns=["0", "1", "2"])
df_bias3 = pd.DataFrame([b3], index=["Threshold"], columns=["0", "1", "2"])

df_layer1 = pd.concat([df_bias1, df_w1])
df_layer3 = pd.concat([df_bias3, df_w3])

print(df_layer1)
print(df_layer3)