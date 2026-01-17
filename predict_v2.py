import pandas as pd
import joblib

model = joblib.load("engagement_drop_model_v2.pkl")
print("✅ Model v2 loaded!")

new_user = pd.DataFrame([{
    "days_active_last_30": 10,
    "last_7_days_active": 1,
    "avg_session_time": 15,
    "videos_watched": 10,
    "support_tickets": 2,
    "tenure_days": 20,
    "payment_failed": 1,
    "plan_type": 0
}])

pred = model.predict(new_user)[0]

if pred == 1:
    print("📌 Prediction: Engagement WILL DROP ❌")
else:
    print("📌 Prediction: Engagement will NOT drop ✅")
