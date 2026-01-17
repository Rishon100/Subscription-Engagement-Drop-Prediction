import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

n = 500

# Basic engagement
days_active_last_30 = np.random.randint(0, 31, n)
last_7_days_active = np.random.randint(0, 8, n)
avg_session_time = np.random.randint(1, 61, n)
videos_watched = np.random.randint(0, 101, n)
support_tickets = np.random.randint(0, 6, n)

# Subscription info
tenure_days = np.random.randint(1, 366, n)  # 1 to 365 days
payment_failed = np.random.randint(0, 2, n)  # 0 or 1

# Plan type: 0=Basic, 1=Standard, 2=Premium
plan_type = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])

# Create a drop risk score (realistic logic)
drop_risk_score = (
    (30 - days_active_last_30) * 0.08 +
    (7 - last_7_days_active) * 0.25 +
    (60 - avg_session_time) * 0.02 +
    (100 - videos_watched) * 0.01 +
    support_tickets * 0.6 +
    payment_failed * 2.0 +
    (2 - plan_type) * 0.3 +      # basic users slightly higher risk
    (50 - tenure_days) * 0.01     # new users higher risk
)

# Convert score into binary target
engagement_drop = (drop_risk_score > 5.5).astype(int)

df = pd.DataFrame({
    "days_active_last_30": days_active_last_30,
    "last_7_days_active": last_7_days_active,
    "avg_session_time": avg_session_time,
    "videos_watched": videos_watched,
    "support_tickets": support_tickets,
    "tenure_days": tenure_days,
    "payment_failed": payment_failed,
    "plan_type": plan_type,
    "engagement_drop": engagement_drop
})

print("✅ New realistic dataset created!")
print(df.head())

print("\nDrop counts:")
print(df["engagement_drop"].value_counts())

# ---------------------------
# ML Training
# ---------------------------
X = df.drop("engagement_drop", axis=1)
y = df["engagement_drop"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n==============================")
print("✅ Random Forest Model Training")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n✅ Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\n✅ Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\n✅ Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\n==============================")
print("✅ Logistic Regression Feature Importance")
print("==============================")

feature_names = X.columns
coefficients = model.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print(importance_df)

import joblib

joblib.dump(model, "engagement_drop_model_v2.pkl")
print("\n✅ Saved: engagement_drop_model_v2.pkl")
