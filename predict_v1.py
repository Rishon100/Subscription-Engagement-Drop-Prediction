import joblib
import pandas as pd

# Load the trained model
model = joblib.load("engagement_drop_model_v1.pkl")
print("✅ Model loaded!")

# New user data for prediction
new_user = pd.DataFrame([{
    "days_active_last_30": 20,
    "avg_session_time": 45,
    "videos_watched": 80,
    "support_tickets": 0
}])

prediction = model.predict(new_user)[0]

if prediction == 1:
    print("📌 Prediction: Engagement WILL DROP ❌")
else:
    print("📌 Prediction: Engagement will NOT drop ✅")