# 📉 Subscription Engagement Drop Prediction (End-to-End ML Project)

This project predicts whether a subscription user's engagement is likely to **drop** based on their recent activity and subscription-related behavior.

## 🌐 Live Demo
🔗 Streamlit App: https://subscription-engagement-drop-predictiongit-jsuwcxe5cjcnfoaha8h.streamlit.app/

✅ Output:
- ✅ Engagement will **NOT** drop (Safe User)
- ❌ Engagement **WILL** drop (High Risk User)

It also shows **Drop Risk %** using model probability.

---

## 🎯 Problem Statement

Subscription-based platforms (OTT, learning apps, fitness apps, etc.) often face a common issue:

📌 Some users slowly stop using the app (lower activity, shorter sessions, complaints, payment issues).  
If we can predict this early, companies can take action like:
- sending reminders
- offering discounts
- improving recommendations
- fixing payment issues

✅ This project predicts:  
**Will the user's engagement drop? (Yes/No)**

---

## ✅ ML Type

- **Supervised Learning**
- **Binary Classification**
- Model predicts:
  - `1` → Engagement Drop
  - `0` → No Engagement Drop

---

## 📊 Features Used

The model is trained using these features:

| Feature Name | Meaning |
|------------|---------|
| `days_active_last_30` | Days user was active in last 30 days |
| `last_7_days_active` | Days active in last 7 days |
| `avg_session_time` | Average session duration (minutes) |
| `videos_watched` | Total videos/content watched |
| `support_tickets` | Number of complaints/issues raised |
| `tenure_days` | Subscription duration in days |
| `payment_failed` | Payment failure status (0/1) |
| `plan_type` | Subscription plan (0=Basic, 1=Standard, 2=Premium) |

---

## 🧠 Models Used

✅ Trained and compared:

### 1) Logistic Regression (Best)
- High accuracy on test dataset
- Works well for linear decision boundaries

### 2) Random Forest (Compared)
- Used for comparison
- Logistic Regression performed better for this dataset

---

## ✅ Model Output Explanation (Simple)

- **Drop Risk %**: probability that the user will drop engagement  
- **Safe Probability %**: probability that the user will NOT drop engagement  

Example:
- Drop Risk = 7%
- Safe Probability = 93%

---

## 🖥️ Streamlit App Features

✅ Built a user-friendly Streamlit app that allows:

- Entering user details using sliders and dropdowns
- Predicting engagement drop instantly
- Showing Drop Risk % with a progress bar
- Displaying clean input summary without horizontal scrolling

---

## 🧠 Explainability (SHAP)
This project uses **SHAP (SHapley Additive exPlanations)** to show the **top reasons behind each prediction**, helping users understand why engagement drop risk is high or low.

## 📁 Project Structure

```txt
SubscriptionDropPrediction/
│
├── app.py                    # Streamlit app
│
├── train_v1.py               # Basic dataset + training
├── train_v2.py               # Improved dataset + training (Best model)
│
├── predict_v1.py             # Predict using v1 model
├── predict_v2.py             # Predict using v2 model
│
├── models/
│   ├── engagement_drop_model_v1.pkl
│   └── engagement_drop_model_v2.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Installation & Setup (Run Locally)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train the model (Version 2 recommended)
python train_v2.py

# 3) Run the Streamlit app
streamlit run app.py
```
