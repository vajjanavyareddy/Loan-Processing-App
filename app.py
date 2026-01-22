import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# --------------------------------------------------
# DARK MODE TOGGLE
# --------------------------------------------------
dark_mode = st.toggle("🌙 Dark Mode")

# Colors
page_bg = "#0E1117" if dark_mode else "#7F7FD5"
card_bg = "#161B22" if dark_mode else "#FFFFFF"
text = "#FFFFFF" if dark_mode else "#111827"

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown(f"""
<style>
/* Page background */
.stApp {{
    background: linear-gradient(135deg, #7F7FD5, #86A8E7, #91EAE4);
    color: {text};
}}

/* Full-width header */
.full-width {{
    width: 100%;
    padding: 3rem 0 4rem 0;
    text-align: center;
}}
.header-inner h1 {{
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0.5rem;
}}
.header-inner p {{
    font-size: 18px;
    opacity: 0.9;
    margin-bottom: 2rem;
}}

/* Stats cards */
.stats-container {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}}
.stat-card {{
    background-color: rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    color: white;
    min-width: 180px;
    text-align: center;
    backdrop-filter: blur(10px);
}}
.stat-card h3 {{ margin: 0; font-size: 20px; }}
.stat-card p {{ margin: 0; font-size: 14px; opacity: 0.9; }}

/* Input & result cards */
.card {{
    background-color: {card_bg};
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}}

/* Result boxes */
.result-approve {{
    background-color: #1E7F4F;
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 700;
    color: white;
    text-align: center;
}}
.result-reject {{
    background-color: #8B1E1E;
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 700;
    color: white;
    text-align: center;
}}

/* Progress bar color */
.stProgress > div > div {{
    background-color: #2EBF91;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER + STATS CARDS
# --------------------------------------------------
st.markdown("""
<div class="full-width">
    <div class="header-inner">
        <h1>💳 Smart Loan Approval System</h1>
        <p>SVM-based Intelligent Loan Decision Platform</p>
        <div class="stats-container">
            <div class="stat-card">
                <h3>ML Powered</h3>
                <p>Support Vector Machines</p>
            </div>
            <div class="stat-card">
                <h3>Real-Time</h3>
                <p>Instant Prediction</p>
            </div>
            <div class="stat-card">
                <h3>Risk Aware</h3>
                <p>Credit-Driven Decisions</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TRAIN MODELS (embedded, no CSV upload)
# --------------------------------------------------
@st.cache_data
def train_models():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
  # use your training dataset

    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df[['Credit_History','LoanAmount','ApplicantIncome','Loan_Amount_Term','Education']]
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Linear SVM": SVC(kernel='linear', probability=True),
        "Polynomial SVM": SVC(kernel='poly', degree=3, probability=True),
        "RBF SVM": SVC(kernel='rbf', probability=True)
    }

    acc = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc[name] = accuracy_score(y_test, model.predict(X_test))

    best_model = max(acc, key=acc.get)
    return models, scaler, acc, best_model

models, scaler, acc, best_kernel = train_models()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Applicant Details")

income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (days)", value=360)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])

kernel = st.radio(
    "Select SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"],
    index=["Linear SVM","Polynomial SVM","RBF SVM"].index(best_kernel)
)

st.caption(f"⭐ Best Performing Kernel: **{best_kernel}**")
st.markdown("</div>", unsafe_allow_html=True)

credit_history = 1 if credit_history == "Good" else 0
education = 1 if education == "Graduate" else 0

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 Check Loan Eligibility", use_container_width=True):
    data = np.array([[credit_history, loan_amount, income, loan_term, education]])
    data = scaler.transform(data)

    model = models[kernel]
    result = model.predict(data)[0]
    confidence = model.predict_proba(data).max()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if result == 1:
        st.markdown("<div class='result-approve'>✅ Loan Approved</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-reject'>❌ Loan Rejected</div>", unsafe_allow_html=True)

    st.write("**Model Confidence**")
    st.progress(int(confidence * 100))
    st.write(f"Confidence Score: **{confidence:.2f}**")
    
    st.subheader("📌 Business Explanation")
    st.write(
        "The decision is based on **credit history, income stability, and loan size**, "
        "indicating the applicant’s **repayment capability**."
    )
    st.markdown("</div>", unsafe_allow_html=True)
