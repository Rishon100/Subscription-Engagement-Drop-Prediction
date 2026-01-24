import numpy as np
import pandas as pd
import joblib
import shap

# Load model
model = joblib.load("models/engagement_drop_model_v2.pkl")
print("✅ Model loaded!")

# -----------------------------
# ✅ Create background data
# (same way we created dataset in train_v2)
# -----------------------------
np.random.seed(42)
n = 300

days_active_last_30 = np.random.randint(0, 31, n)
last_7_days_active = np.random.randint(0, 8, n)
avg_session_time = np.random.randint(1, 61, n)
videos_watched = np.random.randint(0, 101, n)
support_tickets = np.random.randint(0, 6, n)
tenure_days = np.random.randint(1, 366, n)
payment_failed = np.random.randint(0, 2, n)
plan_type = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])

background = pd.DataFrame({
    "days_active_last_30": days_active_last_30,
    "last_7_days_active": last_7_days_active,
    "avg_session_time": avg_session_time,
    "videos_watched": videos_watched,
    "support_tickets": support_tickets,
    "tenure_days": tenure_days,
    "payment_failed": payment_failed,
    "plan_type": plan_type
})

# -----------------------------
# ✅ User to explain
# -----------------------------
new_user = pd.DataFrame([{
    "days_active_last_30": 8,
    "last_7_days_active": 2,
    "avg_session_time": 12,
    "videos_watched": 15,
    "support_tickets": 3,
    "tenure_days": 25,
    "payment_failed": 1,
    "plan_type": 0
}])

# Prediction
prediction = model.predict(new_user)[0]
probability = model.predict_proba(new_user)[0][1]

print("\n📌 Prediction:", "DROP ❌" if prediction == 1 else "NOT DROP ✅")
print(f"📌 Drop Probability: {probability * 100:.2f}%")

# -----------------------------
# ✅ SHAP explain using background
# -----------------------------
explainer = shap.Explainer(model, background)
shap_values = explainer(new_user)

print("\n✅ SHAP Explanation (Feature Contributions):")
shap_result = []

for feature, shap_val in zip(new_user.columns, shap_values.values[0]):
    shap_result.append((feature, shap_val))

# Sort by absolute impact
shap_result = sorted(shap_result, key=lambda x: abs(x[1]), reverse=True)

for feature, shap_val in shap_result:
    direction = "⬆️ increases drop risk" if shap_val > 0 else "⬇️ decreases drop risk"
    print(f"{feature}: {shap_val:.4f}  {direction}")
