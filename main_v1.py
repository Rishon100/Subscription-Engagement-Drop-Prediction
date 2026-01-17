import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix random results
np.random.seed(42)

# Number of users
n = 300

# Create fake user engagement data
days_active_last_30 = np.random.randint(0, 31, n)
avg_session_time = np.random.randint(1, 61, n)
videos_watched = np.random.randint(0, 101, n)
support_tickets = np.random.randint(0, 6, n)

# Target rule
engagement_drop = (
    (days_active_last_30 < 8) |
    (videos_watched < 20) |
    (support_tickets >= 3)
).astype(int)

# DataFrame
df = pd.DataFrame({
    "days_active_last_30": days_active_last_30,
    "avg_session_time": avg_session_time,
    "videos_watched": videos_watched,
    "support_tickets": support_tickets,
    "engagement_drop": engagement_drop
})

print("✅ Dataset created!")
print(df.head())

# -------------------------
# ✅ ML PART STARTS HERE
# -------------------------

# 1) Features (X) and Target (y)
X = df.drop("engagement_drop", axis=1)
y = df["engagement_drop"]

# 2) Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n✅ Train-test split done!")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# 3) Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained!")

# 4) Predictions
y_pred = model.predict(X_test)

# 5) Evaluation
acc = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", acc)

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# 6) Test with a new user
# print("\n✅ Test with a NEW user:")

# new_user = pd.DataFrame([{
#     "days_active_last_30": 3,
#     "avg_session_time": 10,
#     "videos_watched": 5,
#     "support_tickets": 4
# }])

# prediction = model.predict(new_user)[0]

# if prediction == 1:
#     print("📌 Prediction: Engagement WILL DROP ❌")
# else:
#     print("📌 Prediction: Engagement will NOT drop ✅")

import joblib

joblib.dump(model, "engagement_drop_model.pkl")
print("\n✅ Model saved as engagement_drop_model_v1.pkl")
