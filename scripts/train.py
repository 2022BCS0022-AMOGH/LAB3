import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# ------------------ BASE MODEL ------------------
base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# ------------------ HYPERPARAMETER SEARCH SPACE ------------------
param_dist = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}

# ------------------ RANDOMIZED SEARCH ------------------
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=25,                     # enough to beat fixed params
    scoring="r2",                  # primary metric
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# ------------------ BEST MODEL ------------------
model = search.best_estimator_

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
print("XGBOOST + HYPERPARAMETER TUNING RUN")
print(f"Best params: {search.best_params_}")
print(f"R2 Score: {r2}")
print(f"MSE: {mse}")
