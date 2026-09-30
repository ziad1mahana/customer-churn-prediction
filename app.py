import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("src/customer_churn_model.pkl")

# Load feature importance
feature_importance = pd.read_csv(
    "src/feature_importance.csv"
)

# =========================
# Header
# =========================

st.title("📊 Customer Churn Intelligence")

st.write(
    "An AI-powered system for predicting customer churn risk "
    "and supporting data-driven retention decisions."
)

st.divider()

# =========================
# Model Performance
# =========================

st.subheader("📈 Model Performance")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", "80.55%")
m2.metric("Precision", "65.72%")
m3.metric("Recall", "55.88%")
m4.metric("F1 Score", "60.40%")
m5.metric("ROC-AUC", "84.19%")

st.divider()

# =========================
# Customer Profile
# =========================

st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(monthly_charges * tenure)
    )

st.divider()

# =========================
# Prediction
# =========================

if st.button(
    "🔮 Analyze Customer Churn Risk",
    use_container_width=True
):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(
        customer
    )[0, 1]

    # =========================
    # Prediction Result
    # =========================

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if probability >= 0.70:

            st.error("🚨 High Churn Risk")

        elif probability >= 0.40:

            st.warning("⚠️ Medium Churn Risk")

        else:

            st.success("✅ Low Churn Risk")

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.progress(float(probability))

    # =========================
    # Churn Drivers
    # =========================

    st.divider()

    st.subheader("🔎 Top Churn Drivers")

    top_features = feature_importance.head(10).copy()

    top_features = top_features.sort_values(
        "Abs_Coefficient"
    )

    st.bar_chart(
        top_features.set_index("Feature")[
            "Abs_Coefficient"
        ]
    )

    st.caption(
        "Features are ranked by their absolute model coefficient. "
        "This indicates model influence, not a causal relationship."
    )

    # =========================
    # Business Interpretation
    # =========================

    st.divider()

    st.subheader("💡 Business Interpretation")

    if probability >= 0.70:

        st.write(
            "This customer has a high predicted churn probability. "
            "The business may consider reviewing the customer's "
            "contract, pricing, service usage, and support experience."
        )

    elif probability >= 0.40:

        st.write(
            "This customer shows a moderate predicted churn risk. "
            "Monitoring the customer's engagement and service experience "
            "may help identify potential retention opportunities."
        )

    else:

        st.write(
            "This customer has a lower predicted churn probability "
            "according to the model."
        )

    # =========================
    # Customer Information
    # =========================

    st.divider()

    st.subheader("👤 Customer Information")

    st.dataframe(
        customer,
        use_container_width=True
    )