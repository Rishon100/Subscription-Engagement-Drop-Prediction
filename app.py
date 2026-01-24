import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("models/engagement_drop_model_v2.pkl")

st.set_page_config(page_title="Engagement Drop Prediction", page_icon="📉", layout="wide")

st.title("📉 Subscription Engagement Drop Prediction")
st.write("Enter user details and predict if engagement will drop.")
st.caption("✅ Includes prediction + drop risk % + top reasons (Explainability using SHAP)")

st.sidebar.header("🧾 User Inputs")

# -----------------------------
# User Inputs
# -----------------------------
days_active_last_30 = st.sidebar.slider("Days Active (Last 30 Days)", 0, 30, 10)
last_7_days_active = st.sidebar.slider("Days Active (Last 7 Days)", 0, 7, 2)
avg_session_time = st.sidebar.slider("Avg Session Time (minutes)", 1, 60, 15)
videos_watched = st.sidebar.slider("Videos Watched", 0, 100, 20)
support_tickets = st.sidebar.slider("Support Tickets", 0, 5, 1)
tenure_days = st.sidebar.slider("Tenure Days", 1, 365, 30)

payment_failed_text = st.sidebar.selectbox("Payment Failed?", ["No", "Yes"])
payment_failed = 1 if payment_failed_text == "Yes" else 0

plan_text = st.sidebar.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
plan_type = plan_map[plan_text]

# Model input dataframe
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

# ✅ Display input nicely
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

st.dataframe(pretty_input, use_container_width=True)

# -----------------------------
# Background data for SHAP
# (We create synthetic background similar to training)
# -----------------------------
np.random.seed(42)
n_bg = 300

background = pd.DataFrame({
    "days_active_last_30": np.random.randint(0, 31, n_bg),
    "last_7_days_active": np.random.randint(0, 8, n_bg),
    "avg_session_time": np.random.randint(1, 61, n_bg),
    "videos_watched": np.random.randint(0, 101, n_bg),
    "support_tickets": np.random.randint(0, 6, n_bg),
    "tenure_days": np.random.randint(1, 366, n_bg),
    "payment_failed": np.random.randint(0, 2, n_bg),
    "plan_type": np.random.choice([0, 1, 2], size=n_bg, p=[0.5, 0.3, 0.2])
})

# -----------------------------
# Prediction + Explainability
# -----------------------------
if st.button("Predict Engagement Drop"):
    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    prob_not_drop = proba[0] * 100
    prob_drop = proba[1] * 100

    st.subheader("📊 Drop Risk Percentage")

    st.write(f"❌ Drop Risk: **{prob_drop:.2f}%**")
    st.progress(int(prob_drop))

    st.write(f"✅ Safe Probability (NOT Drop): **{prob_not_drop:.2f}%**")

    # Final label
    if prediction == 1:
        st.error("❌ Engagement WILL DROP (High Risk User)")
    else:
        st.success("✅ Engagement will NOT Drop (Safe User)")

    # -----------------------------
    # ✅ SHAP Explanation (Top Reasons)
    # -----------------------------
    st.subheader("🧠 Top Reasons (Explainability)")

    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_data)

    # Get SHAP values for the one row
    shap_list = []
    for feature, val in zip(input_data.columns, shap_values.values[0]):
        shap_list.append((feature, val))

    # Sort by absolute value (most impact first)
    shap_list = sorted(shap_list, key=lambda x: abs(x[1]), reverse=True)

    # Show top 3 reasons
    top_k = 3
    for i in range(top_k):
        feature, val = shap_list[i]

        if val > 0:
            st.write(f"✅ **{feature}** increased drop risk (SHAP: {val:.3f})")
        else:
            st.write(f"✅ **{feature}** decreased drop risk (SHAP: {val:.3f})")
