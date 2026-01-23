import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score

# ------------------ LOAD DATA ------------------
DATA_PATH = "dataset/winequality-red.csv"

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ PREDICTION ------------------
y_pred = model.predict(X_test)

# ------------------ METRICS ------------------
# Regression metric
mse = mean_squared_error(y_test, y_pred)

# Convert regression output to binary classification for F1
# Good wine: quality >= 6
y_test_bin = (y_test >= 6).astype(int)
y_pred_bin = (y_pred >= 6).astype(int)

f1 = f1_score(y_test_bin, y_pred_bin)

# ------------------ SAVE MODEL ------------------
joblib.dump(model, "model.pkl")

# ------------------ SAVE METRICS ------------------
metrics = {
    "f1": float(f1),
    "mse": float(mse)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------ LOGS ------------------
print("Training complete")
print(f"F1 Score: {f1}")
print(f"MSE: {mse}")
