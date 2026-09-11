import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - PROFESSIONAL UI
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Remove extra top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 120px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 700;
    }

    /* Information cards */
    .info-card {
        background: white;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* Prediction result */
    .prediction-card {
        background: white;
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }

    .prediction-probability {
        font-size: 42px;
        font-weight: 800;
        margin: 10px 0;
    }

    /* Small graph container */
    .graph-card {
        background: white;
        padding: 12px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.55rem 1rem;
    }

    /* Divider */
    hr {
        margin-top: 25px;
        margin-bottom: 25px;
        border-color: #e5e7eb;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, feature_columns


model, scaler, feature_columns = load_model_files()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

    return data


df = load_dataset()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 25px 0;">
            <div style="font-size:42px;">📊</div>
            <div style="font-size:22px; font-weight:700;">
                Churn Analytics
            </div>
            <div style="font-size:13px; color:#9ca3af;">
                Machine Learning System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔮 Predict Churn",
            "📊 Dashboard",
            "🤖 Model Analysis"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size:12px; color:#9ca3af; line-height:1.6;">
        <b>Project:</b><br>
        Customer Churn Prediction<br><br>

        <b>Algorithm:</b><br>
        Logistic Regression<br><br>

        <b>Dataset:</b><br>
        IBM Telco Customer Churn
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'A machine learning system for predicting customer churn '
        'and identifying high-risk customers.'
        '</div>',
        unsafe_allow_html=True
    )

    # Overview cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-value">7,043</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Churned Customers</div>
                <div class="metric-value">1,869</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-value">26.54%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model ROC-AUC</div>
                <div class="metric-value">84.16%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # About project

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.markdown(
            """
            <div class="info-card">

            <h3>🎯 Project Objective</h3>

            The objective of this project is to develop an
            end-to-end machine learning application that predicts
            whether a telecom customer is likely to churn.

            <br><br>

            The system analyzes customer demographic information,
            account information, services and billing details to
            estimate churn probability.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

            <h3>🤖 Final Model</h3>

            <b>Algorithm:</b> Logistic Regression<br><br>

            <b>Accuracy:</b> 80.70%<br>
            <b>Precision:</b> 65.84%<br>
            <b>Recall:</b> 56.68%<br>
            <b>F1 Score:</b> 60.92%<br>
            <b>ROC-AUC:</b> 84.16%

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🔄 Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <b>1. Dataset Collection</b> → 
        <b>2. Data Preprocessing</b> → 
        <b>3. EDA</b> → 
        <b>4. Feature Engineering</b> → 
        <b>5. Data Splitting</b> → 
        <b>6. Model Training</b> → 
        <b>7. Evaluation</b> → 
        <b>8. Prediction</b>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICT CHURN
# =========================================================

elif page == "🔮 Predict Churn":

    st.markdown(
        '<div class="main-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter customer information to estimate churn probability.'
        '</div>',
        unsafe_allow_html=True
    )

    # Customer information

    st.markdown(
        '<div class="section-title">👤 Customer Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
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
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:

        multiple = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=monthly * max(tenure, 1)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone],
            "MultipleLines": [multiple],
            "InternetService": [internet],
            "OnlineSecurity": [security],
            "OnlineBackup": [backup],
            "DeviceProtection": [device],
            "TechSupport": [tech],
            "StreamingTV": [tv],
            "StreamingMovies": [movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total]
        })

        encoded_data = pd.get_dummies(
            input_data,
            drop_first=True
        )

        encoded_data = encoded_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        scaled_data = scaler.transform(
            encoded_data
        )

        prediction = model.predict(
            scaled_data
        )[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1]

        probability_percent = probability * 100

        if probability_percent >= 70:

            risk = "HIGH RISK"
            recommendation = (
                "Immediate customer retention action is recommended."
            )

        elif probability_percent >= 40:

            risk = "MEDIUM RISK"
            recommendation = (
                "Customer should be monitored and offered suitable incentives."
            )

        else:

            risk = "LOW RISK"
            recommendation = (
                "Customer currently shows a relatively low churn risk."
            )

        st.markdown("---")

        if prediction == 1:

            result_text = "⚠️ Customer likely to churn"

        else:

            result_text = "✅ Customer likely to stay"

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    {result_text}
                </div>

                <div class="prediction-probability">
                    {probability_percent:.2f}%
                </div>

                <div style="color:#6b7280;">
                    Estimated Churn Probability
                </div>

                <br>

                <b>Risk Level:</b> {risk}

                <br><br>

                <div style="color:#4b5563;">
                    {recommendation}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            int(probability_percent)
        )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">Customer Churn Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore customer behavior and churn patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    total_customers = len(df)

    churned = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned / total_customers
    ) * 100

    non_churn_rate = 100 - churn_rate

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-value">
                    {total_customers:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Churned Customers</div>
                <div class="metric-value">
                    {churned:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Churn Rate</div>
                <div class="metric-value">
                    {churn_rate:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Non-Churn Rate</div>
                <div class="metric-value">
                    {non_churn_rate:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # GRAPH 1 + GRAPH 2
    # =====================================================

    col1, col2 = st.columns(2)

    # Churn distribution

    with col1:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(4.8, 3.0)
        )

        churn_counts = df["Churn"].value_counts()

        sns.barplot(
            x=churn_counts.index,
            y=churn_counts.values,
            ax=ax
        )

        ax.set_title(
            "Customer Churn Distribution",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Customers")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Contract

    with col2:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(4.8, 3.0)
        )

        sns.countplot(
            data=df,
            x="Contract",
            hue="Churn",
            ax=ax
        )

        ax.set_title(
            "Contract Type vs Churn",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Customers")

        ax.tick_params(
            axis="x",
            rotation=10
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # GRAPH 3 + GRAPH 4
    # =====================================================

    col1, col2 = st.columns(2)

    # Tenure

    with col1:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(4.8, 3.0)
        )

        sns.boxplot(
            data=df,
            x="Churn",
            y="tenure",
            ax=ax
        )

        ax.set_title(
            "Tenure vs Churn",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Tenure (Months)")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Monthly Charges

    with col2:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(4.8, 3.0)
        )

        sns.boxplot(
            data=df,
            x="Churn",
            y="MonthlyCharges",
            ax=ax
        )

        ax.set_title(
            "Monthly Charges vs Churn",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Monthly Charges")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # GRAPH 5
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(4.8, 3.0)
        )

        sns.countplot(
            data=df,
            x="InternetService",
            hue="Churn",
            ax=ax
        )

        ax.set_title(
            "Internet Service vs Churn",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Customers")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MODEL ANALYSIS
# =========================================================

elif page == "🤖 Model Analysis":

    st.markdown(
        '<div class="main-title">Model Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Performance comparison and evaluation of the trained models.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # MODEL COMPARISON TABLE
    # =====================================================

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree"
        ],

        "Accuracy": [
            80.70,
            79.21,
            79.42
        ],

        "Precision": [
            65.84,
            69.12,
            62.96
        ],

        "Recall": [
            56.68,
            49.73,
            54.55
        ],

        "F1 Score": [
            60.92,
            55.94,
            58.45
        ],

        "ROC-AUC": [
            84.16,
            82.59,
            82.84
        ]
    })

    st.markdown(
        '<div class="section-title">📋 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # MODEL GRAPH
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Model Comparison</div>',
        unsafe_allow_html=True
    )

    col_left, col_graph, col_right = st.columns(
        [1, 2, 1]
    )

    with col_graph:

        fig, ax = plt.subplots(
            figsize=(5.2, 3.2)
        )

        x = np.arange(
            len(results["Model"])
        )

        width = 0.16

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]

        for i, metric in enumerate(metrics):

            ax.bar(
                x + (i - 2) * width,
                results[metric],
                width,
                label=metric
            )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                "Logistic\nRegression",
                "Random\nForest",
                "Decision\nTree"
            ],
            fontsize=9
        )

        ax.set_ylabel("Percentage")

        ax.set_title(
            "Model Performance Comparison",
            fontsize=12,
            fontweight="bold"
        )

        ax.legend(
            fontsize=7,
            ncol=2
        )

        ax.set_ylim(0, 100)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

    # =====================================================
    # TEST DATA FOR CONFUSION MATRIX + ROC
    # =====================================================

    analysis_df = df.drop(
        "customerID",
        axis=1
    )

    X_analysis = analysis_df.drop(
        "Churn",
        axis=1
    )

    y_analysis = analysis_df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X_analysis = pd.get_dummies(
        X_analysis,
        drop_first=True
    )

    X_train_analysis, X_test_analysis, y_train_analysis, y_test_analysis = train_test_split(
        X_analysis,
        y_analysis,
        test_size=0.20,
        random_state=42,
        stratify=y_analysis
    )

    X_test_analysis = X_test_analysis.reindex(
        columns=feature_columns,
        fill_value=0
    )

    X_test_analysis_scaled = scaler.transform(
        X_test_analysis
    )

    logistic_predictions = model.predict(
        X_test_analysis_scaled
    )

    logistic_probabilities = model.predict_proba(
        X_test_analysis_scaled
    )[:, 1]

    # =====================================================
    # CONFUSION MATRIX + ROC
    # =====================================================

    col1, col2 = st.columns(2)

    # Confusion matrix

    with col1:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test_analysis,
            logistic_predictions
        )

        fig, ax = plt.subplots(
            figsize=(3.8, 3.0)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticklabels(
            ["No Churn", "Churn"]
        )

        ax.set_yticklabels(
            ["No Churn", "Churn"],
            rotation=0
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ROC curve

    with col2:

        st.markdown(
            '<div class="graph-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### ROC Curve"
        )

        fpr, tpr, thresholds = roc_curve(
            y_test_analysis,
            logistic_probabilities
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        fig, ax = plt.subplots(
            figsize=(3.8, 3.0)
        )

        ax.plot(
            fpr,
            tpr,
            label=f"Logistic Regression (AUC = {roc_auc:.2f})"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve",
            fontsize=12,
            fontweight="bold"
        )

        ax.legend(
            fontsize=8
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # FINAL MODEL
    # =====================================================

    st.markdown(
        '<div class="section-title">🏆 Final Model Selection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>Logistic Regression</h3>

        Logistic Regression was selected as the final model because
        it achieved the strongest overall performance among the
        evaluated models.

        <br><br>

        <b>Accuracy:</b> 80.70% &nbsp;&nbsp;
        <b>Recall:</b> 56.68% &nbsp;&nbsp;
        <b>F1 Score:</b> 60.92% &nbsp;&nbsp;
        <b>ROC-AUC:</b> 84.16%

        <br><br>

        The ROC-AUC score of 84.16% indicates that the model has
        good ability to distinguish between customers who churn
        and customers who stay.

        </div>
        """,
        unsafe_allow_html=True
    )