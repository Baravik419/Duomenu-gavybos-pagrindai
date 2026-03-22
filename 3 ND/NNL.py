from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)