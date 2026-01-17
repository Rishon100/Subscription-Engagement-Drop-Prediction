import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/engagement_drop_model_v2.pkl")

st.set_page_config(page_title="Engagement Drop Prediction", page_icon="📉", layout="wide")

st.title("📉 Subscription Engagement Drop Prediction")
st.write("Enter user details and predict if engagement will drop.")

st.sidebar.header("🧾 User Inputs")

# Numeric inputs
days_active_last_30 = st.sidebar.slider("Days Active (Last 30 Days)", 0, 30, 10)
last_7_days_active = st.sidebar.slider("Days Active (Last 7 Days)", 0, 7, 2)
avg_session_time = st.sidebar.slider("Avg Session Time (minutes)", 1, 60, 15)
videos_watched = st.sidebar.slider("Videos Watched", 0, 100, 20)
support_tickets = st.sidebar.slider("Support Tickets", 0, 5, 1)
tenure_days = st.sidebar.slider("Tenure Days", 1, 365, 30)

# User-friendly Payment Failed
payment_failed_text = st.sidebar.selectbox("Payment Failed?", ["No", "Yes"])
payment_failed = 1 if payment_failed_text == "Yes" else 0

# User-friendly Plan Type
plan_text = st.sidebar.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
plan_type = plan_map[plan_text]

# Create input dataframe (exact feature names used in training)
input_data = pd.DataFrame([{
    "days_active_last_30": days_active_last_30,
    "last_7_days_active": last_7_days_active,
    "avg_session_time": avg_session_time,
    "videos_watched": videos_watched,
    "support_tickets": support_tickets,
    "tenure_days": tenure_days,
    "payment_failed": payment_failed,
    "plan_type": plan_type
}])

# ✅ Display input nicely (no left-right scrolling)
st.subheader("✅ User Input Details")

pretty_input = pd.DataFrame([{
    "Days Active (30d)": days_active_last_30,
    "Days Active (7d)": last_7_days_active,
    "Avg Session Time (min)": avg_session_time,
    "Videos Watched": videos_watched,
    "Support Tickets": support_tickets,
    "Tenure Days": tenure_days,
    "Payment Failed?": payment_failed_text,
    "Plan Type": plan_text
}])

# Use container_width so it fits full screen (no scroll)
st.dataframe(pretty_input, use_container_width=True)

# Prediction
if st.button("Predict Engagement Drop"):
    prediction = model.predict(input_data)[0]

    # Probability output
    proba = model.predict_proba(input_data)[0]
    prob_not_drop = proba[0] * 100
    prob_drop = proba[1] * 100

    st.subheader("📊 Drop Risk Percentage")

    # ✅ Progress bar (0 to 100)
    st.write(f"❌ Drop Risk: **{prob_drop:.2f}%**")
    st.progress(int(prob_drop))

    st.write(f"✅ Safe Probability (NOT Drop): **{prob_not_drop:.2f}%**")

    # Final result
    if prediction == 1:
        st.error("❌ Engagement WILL DROP (High Risk User)")
    else:
        st.success("✅ Engagement will NOT Drop (Safe User)")
