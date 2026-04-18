import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# -------------------------------
# Load Data
# -------------------------------
file_path = r"D:\Vedant's Stuff\BDA\IA\data\raw\heart_disease_uci.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

# -------------------------------
# Target Handling
# -------------------------------
target_col = "num"
df[target_col] = (df[target_col] > 0).astype(int)

# -------------------------------
# Feature Engineering (HIGH IMPACT)
# -------------------------------
df["chol_age_ratio"] = df["chol"] / df["age"]
df["thalach_age_ratio"] = df["thalach"] / df["age"]
df["oldpeak_sq"] = df["oldpeak"] ** 2

df["high_bp"] = (df["trestbps"] > 140).astype(int)
df["high_chol"] = (df["chol"] > 240).astype(int)
df["low_hr"] = (df["thalach"] < 120).astype(int)

df["age_oldpeak"] = df["age"] * df["oldpeak"]
df["chol_oldpeak"] = df["chol"] * df["oldpeak"]

# -------------------------------
# Split FIRST (avoid leakage)
# -------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Imputation (KNN > median)
# -------------------------------
imputer = KNNImputer(n_neighbors=5)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# SMOTE (ONLY on training data)
# -------------------------------
sm = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = sm.fit_resample(X_train, y_train)

# -------------------------------
# XGBoost Model (TUNED)
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_lambda=1,
    reg_alpha=0.5,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))