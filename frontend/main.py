import streamlit as st
import requests
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Credit Risk AI", page_icon="📊", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e2f, #2a5298);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    opacity: 0.7;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* Result Cards */
.result-card {
    background: rgba(0,0,0,0.4);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    border: none;
    border-radius: 8px;
    font-size: 18px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">💳 CREDIT RISK ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered loan risk evaluation system</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 18, 100, 28)
    with col2:
        income = st.number_input('Income', value=1200000)
    with col3:
        loan_amount = st.number_input('Loan Amount', value=2560000)

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        loan_tenure_months = st.number_input('Loan Tenure', value=36)
    with col5:
        avg_dpd_per_delinquency = st.number_input('Avg DPD', value=20)
    with col6:
        num_open_accounts = st.number_input('Open Accounts', value=2)

    # Row 3
    col7, col8, col9 = st.columns(3)
    with col7:
        delinquency_ratio = st.number_input('Delinquency Ratio', value=30)
    with col8:
        credit_utilization_ratio = st.number_input('Credit Utilization', value=30)
    with col9:
        residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])

    # Row 4
    col10, col11 = st.columns(2)
    with col10:
        loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
    with col11:
        loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

    # Ratio
    ratio = loan_amount / income if income > 0 else 0
    st.markdown(f"**Loan to Income Ratio:** `{ratio:.2f}`")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BUTTON ----------------
if st.button("🚀 Calculate Risk"):

    try:
        with st.spinner("Analyzing risk..."):

            response = requests.post(
                f"{API_URL}/predict_credit_risk",
                json={
                    "age": age,
                    "income": income,
                    "loan_amount": loan_amount,
                    "loan_tenure_months": loan_tenure_months,
                    "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
                    "delinquency_ratio": delinquency_ratio,
                    "credit_utilization_ratio": credit_utilization_ratio,
                    "num_open_accounts": num_open_accounts,
                    "residence_type": residence_type,
                    "loan_purpose": loan_purpose,
                    "loan_type": loan_type
                },
                timeout=10
            )

        if response.status_code == 200:
            result = response.json()

            prob = result['probability']
            score = result['credit_score']
            rating = result['rating']
            top_features = result.get('top_features', [])

            # Color logic
            if rating == "Excellent":
                color = "#00ffcc"
            elif rating == "Good":
                color = "#66ff66"
            elif rating == "Average":
                color = "#ffaa00"
            else:
                color = "#ff4d4d"

            # ---------------- RESULTS ----------------
            st.markdown("### 📊 Results")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="result-card">
                <h4>Probability</h4>
                <h2>{prob:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="result-card">
                <h4>Credit Score</h4>
                <h2>{score}</h2>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="result-card">
                <h4>Rating</h4>
                <h2 style="color:{color}">{rating}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🔍 Key Risk Drivers")

            if top_features:
                for feature, impact in top_features:
                    feature_name = feature.replace('_', ' ').title()

                    if impact > 0:
                        st.markdown(f"🔺 **{feature_name}** increasing risk")
                    else:
                        st.markdown(f"🟢 **{feature_name}** reducing risk")
            else:
                st.info("No feature contribution data available.")

        else:
            st.error("API Error ❌")

    except Exception as e:
        st.error(f"Connection Error: {str(e)}")