import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F5F8F8;
    }

    /* Remove excessive Streamlit top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hide Streamlit branding/menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ========================================================
       BRAND HEADER
       ======================================================== */

    .brand-header {
        background: linear-gradient(135deg, #073B3A 0%, #087F7B 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 25px rgba(7, 59, 58, 0.12);
    }

    .brand-name {
        color: #7DE2D1;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .brand-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }

    .brand-description {
        color: #D9F3EF;
        font-size: 0.95rem;
        margin-top: 0.7rem;
        margin-bottom: 0;
        max-width: 650px;
        line-height: 1.6;
    }

    .model-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 0.8rem;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        color: #E9FFFB;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #173F3E;
        margin-top: 0.4rem;
        margin-bottom: 0.15rem;
    }

    .section-description {
        color: #718080;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .input-card {
        background: white;
        padding: 1.4rem 1.5rem 1.1rem 1.5rem;
        border-radius: 16px;
        border: 1px solid #DCE7E6;
        box-shadow: 0 3px 12px rgba(0,0,0,0.035);
        min-height: 100%;
    }

    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 9px !important;
        border-color: #D6E2E1 !important;
        background-color: #FFFFFF !important;
    }

    div[data-baseweb="input"] > div {
        border-radius: 9px !important;
        border-color: #D6E2E1 !important;
        background-color: #FFFFFF !important;
    }

    .stSlider > div > div > div {
        color: #087F7B;
    }

    label {
        font-weight: 500 !important;
        color: #3E5554 !important;
        font-size: 0.85rem !important;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    div.stButton > button {
        background: linear-gradient(135deg, #087F7B, #05A89F);
        color: white;
        border: none;
        border-radius: 11px;
        padding: 0.75rem 1.2rem;
        font-size: 0.95rem;
        font-weight: 700;
        transition: all 0.2s ease;
        box-shadow: 0 5px 15px rgba(8, 127, 123, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #066C69, #078F89);
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(8, 127, 123, 0.25);
    }

    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {
        background: white;
        border: 1px solid #DCE7E6;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 5px 18px rgba(0,0,0,0.045);
    }

    .result-label {
        color: #718080;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-probability {
        color: #173F3E;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }

    .result-description {
        color: #647574;
        font-size: 0.88rem;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        text-align: center;
        color: #8A9A99;
        font-size: 0.72rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #DCE7E6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()

# ============================================================
# BRAND HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-name">ChurnIQ</div>
        <div class="brand-title">Customer Churn Prediction</div>
        <p class="brand-description">
            Estimate a customer's likelihood of leaving the service
            using a machine learning model trained on Telco customer data.
        </p>
        <div class="model-badge">Random Forest • ROC-AUC 0.84</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">Customer Information</div>
    <div class="section-description">
        Enter the customer's account and service details below.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

# ============================================================
# ACCOUNT CARD
# ============================================================

with col1:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">Account Details</div>
        <div class="section-description">
            Customer demographics, tenure and billing information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tenure = st.slider(
        "Tenure (months)",
        0,
        72,
        12,
    )

    monthly_charges = st.slider(
        "Monthly Charges ($)",
        18.0,
        120.0,
        65.0,
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
    )

    partner = st.selectbox(
        "Has Partner",
        ["Yes", "No"],
    )

    dependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"],
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SERVICES CARD
# ============================================================

with col2:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">Services</div>
        <div class="section-description">
            Customer's subscribed telecommunications services.
        </div>
        """,
        unsafe_allow_html=True,
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    if phone_service == "No":
        multiple_lines = "No phone service"
        st.caption("Multiple Lines: No phone service")
    else:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No"],
        )

    internet_service = st.selectbox(
        "Internet Service",
        ["Fiber optic", "DSL", "No"],
    )

    if internet_service == "No":
        addon_options = ["No internet service"]
    else:
        addon_options = ["Yes", "No"]

    online_security = st.selectbox(
        "Online Security",
        addon_options,
    )

    online_backup = st.selectbox(
        "Online Backup",
        addon_options,
    )

    device_protection = st.selectbox(
        "Device Protection",
        addon_options,
    )

    tech_support = st.selectbox(
        "Tech Support",
        addon_options,
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        addon_options,
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        addon_options,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# BUILD MODEL INPUT
# ============================================================

input_df = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
        }
    ]
)

# ============================================================
# PREDICTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col, _ = st.columns([1, 2])

with predict_col:

    if st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True,
    ):

        proba = model.predict_proba(input_df)[0, 1]

        prediction = (
            "Likely to churn"
            if proba >= 0.5
            else "Likely to stay"
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-label">Churn Probability</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-probability">{proba:.1%}</div>',
            unsafe_allow_html=True,
        )

        if proba >= 0.5:

            st.error(
                f"⚠️ {prediction}"
            )

            st.markdown(
                """
                <div class="result-description">
                    This customer is currently classified as having
                    an elevated likelihood of churn.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.success(
                f"✅ {prediction}"
            )

            st.markdown(
                """
                <div class="result-description">
                    This customer is currently classified as having
                    a lower likelihood of churn.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(proba, 1.0))

        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # Model Input
        # ----------------------------------------------------

        with st.expander("View customer profile used for prediction"):

            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        ChurnIQ • Customer Churn Prediction
        <br>
        Random Forest model trained on Telco customer data.
        TotalCharges excluded from the feature set.
    </div>
    """,
    unsafe_allow_html=True,
)