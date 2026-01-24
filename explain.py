import pandas as pd
import joblib
import shap

# Load model
model = joblib.load("models/engagement_drop_model_v2.pkl")
print("✅ Model loaded!")

# Example user input (you can change values)
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

# Predict
prediction = model.predict(new_user)[0]
probability = model.predict_proba(new_user)[0][1]

print("\n📌 Prediction:", "DROP ❌" if prediction == 1 else "NOT DROP ✅")
print(f"📌 Drop Probability: {probability * 100:.2f}%")

# ✅ SHAP Explanation
explainer = shap.Explainer(model, new_user)
shap_values = explainer(new_user)

print("\n✅ SHAP Explanation (Feature Contributions):")
for feature, value in zip(new_user.columns, shap_values.values[0]):
    print(f"{feature}: {value:.4f}")
