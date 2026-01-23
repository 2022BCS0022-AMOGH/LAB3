import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ------------------ LOAD DATA ------------------
DATA_PATH = "dataset/winequality-red.csv"

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ XGBOOST MODEL ------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ SAVE MODEL ------------------
joblib.dump(model, "model.pkl")

# ------------------ SAVE METRICS ------------------
metrics = {
    "r2": float(r2),
    "mse": float(mse)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------ LOGS ------------------
print("XGBOOST TRAINING RUN")
print(f"R2 Score: {r2}")
print(f"MSE: {mse}")
