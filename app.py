import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# SIMPLE PROFESSIONAL CSS
# IMPORTANT: This is CSS only. No HTML UI components.
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f6f8fc;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Navigation buttons */
    .nav-button button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    /* Main buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 650;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 25px;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
    }

    /* Mobile */
    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 21px;
        }

        div[data-testid="stMetric"] {
            padding: 12px;
        }

    }

    @media (max-width: 480px) {

        .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 19px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


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
# LOAD DATA
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
# PAGE STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def change_page(page_name):
    st.session_state.page = page_name


# =========================================================
# HEADER
# =========================================================

title_col, badge_col = st.columns([4, 1])

with title_col:

    st.markdown(
        "## 📊 ChurnIQ"
    )

    st.caption(
        "Customer Churn Prediction & Analytics"
    )

with badge_col:

    st.write("")
    st.success("ML SYSTEM")


# =========================================================
# NAVIGATION
# =========================================================

n1, n2, n3, n4 = st.columns(4)

with n1:
    st.button(
        "🏠 Home",
        use_container_width=True,
        on_click=change_page,
        args=("Home",),
        type="primary"
        if st.session_state.page == "Home"
        else "secondary"
    )

with n2:
    st.button(
        "🔮 Predict",
        use_container_width=True,
        on_click=change_page,
        args=("Predict",),
        type="primary"
        if st.session_state.page == "Predict"
        else "secondary"
    )

with n3:
    st.button(
        "📊 Dashboard",
        use_container_width=True,
        on_click=change_page,
        args=("Dashboard",),
        type="primary"
        if st.session_state.page == "Dashboard"
        else "secondary"
    )

with n4:
    st.button(
        "🤖 Model",
        use_container_width=True,
        on_click=change_page,
        args=("Model",),
        type="primary"
        if st.session_state.page == "Model"
        else "secondary"
    )


st.divider()


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.title("Customer Churn Intelligence")

    st.write(
        "A machine learning application that predicts customer "
        "churn probability and helps identify customers who may "
        "need retention attention."
    )

    st.write("")

    total_customers = len(df)

    churned = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned / total_customers
    ) * 100

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "Churned Customers",
            f"{churned:,}"
        )

    with c3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with c4:
        st.metric(
            "ROC-AUC",
            "84.16%"
        )

    st.write("")

    # -----------------------------------------------------
    # ABOUT
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.subheader("🎯 Project Objective")

            st.write(
                "The system analyzes customer demographic, "
                "service, contract and billing information "
                "to estimate the probability of customer churn."
            )

    with c2:

        with st.container(border=True):

            st.subheader("🤖 Final Model")

            st.write(
                "Logistic Regression was selected as the final "
                "model after comparing Logistic Regression, "
                "Random Forest and Decision Tree."
            )

            st.write(
                "**Accuracy:** 80.70%  •  "
                "**F1:** 60.92%  •  "
                "**ROC-AUC:** 84.16%"
            )

    st.subheader("🔄 Machine Learning Workflow")

    st.info(
        "Dataset → Preprocessing → EDA → Feature Engineering "
        "→ Train/Test Split → Model Training → Evaluation → Prediction"
    )

    st.subheader("✨ Application Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        with st.container(border=True):
            st.write("### 🔮 Prediction")
            st.write(
                "Estimate the probability that a customer will churn."
            )

    with f2:
        with st.container(border=True):
            st.write("### 📊 Analytics")
            st.write(
                "Explore customer churn patterns through visualizations."
            )

    with f3:
        with st.container(border=True):
            st.write("### 🤖 Evaluation")
            st.write(
                "Compare machine learning models using standard metrics."
            )


# =========================================================
# PREDICT
# =========================================================

elif st.session_state.page == "Predict":

    st.title("🔮 Customer Churn Prediction")

    st.caption(
        "Enter customer information and let the trained model "
        "estimate churn probability."
    )

    # -----------------------------------------------------
    # PERSONAL INFORMATION
    # -----------------------------------------------------

    with st.expander(
        "👤 Personal Information",
        expanded=True
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

        with c2:

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with c3:

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

    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    with st.expander(
        "📡 Services",
        expanded=True
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            multiple = st.selectbox(
                "Multiple Lines",
                [
                    "No phone service",
                    "No",
                    "Yes"
                ]
            )

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

        with c2:

            security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with c3:

            device = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tech = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

    # -----------------------------------------------------
    # ACCOUNT INFORMATION
    # -----------------------------------------------------

    with st.expander(
        "💳 Account & Billing",
        expanded=True
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

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

        with c2:

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

        with c3:

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(
                    monthly * max(tenure, 1)
                )
            )

    st.write("")

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    predict = st.button(
        "✨ Analyze Customer Risk",
        use_container_width=True,
        type="primary"
    )

    if predict:

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

        try:

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

            probability_percent = (
                probability * 100
            )

            # Risk classification

            if probability_percent >= 70:

                risk = "HIGH RISK"
                recommendation = (
                    "Immediate customer retention action is recommended."
                )

            elif probability_percent >= 40:

                risk = "MEDIUM RISK"
                recommendation = (
                    "Monitor this customer and consider "
                    "suitable retention incentives."
                )

            else:

                risk = "LOW RISK"
                recommendation = (
                    "Customer currently shows relatively low "
                    "churn risk."
                )

            st.divider()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to churn"
                )

            else:

                st.success(
                    "✅ Customer is likely to stay"
                )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Churn Probability",
                    f"{probability_percent:.2f}%"
                )

            with r2:

                st.metric(
                    "Risk Level",
                    risk
                )

            with r3:

                result = (
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay"
                )

                st.metric(
                    "Prediction",
                    result
                )

            st.subheader("Churn Probability")

            st.progress(
                int(
                    min(
                        max(
                            probability_percent,
                            0
                        ),
                        100
                    )
                )
            )

            st.caption(
                f"Estimated churn probability: "
                f"{probability_percent:.2f}%"
            )

            if probability_percent >= 70:

                st.warning(
                    f"**Recommendation:** {recommendation}"
                )

            elif probability_percent >= 40:

                st.info(
                    f"**Recommendation:** {recommendation}"
                )

            else:

                st.success(
                    f"**Recommendation:** {recommendation}"
                )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# =========================================================
# DASHBOARD
# =========================================================

elif st.session_state.page == "Dashboard":

    st.title("📊 Customer Dashboard")

    st.caption(
        "Explore customer behavior and churn patterns."
    )

    total_customers = len(df)

    churned = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned / total_customers
    ) * 100

    retained_rate = 100 - churn_rate

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "Churned",
            f"{churned:,}"
        )

    with c3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with c4:
        st.metric(
            "Retained",
            f"{retained_rate:.2f}%"
        )

    st.subheader("Customer Insights")

    # -----------------------------------------------------
    # GRAPH 1
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("**Churn Distribution**")

            fig, ax = plt.subplots(
                figsize=(4.2, 2.5)
            )

            counts = df["Churn"].value_counts()

            sns.barplot(
                x=counts.index,
                y=counts.values,
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("Customers")
            ax.tick_params(labelsize=8)

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 2
    # -----------------------------------------------------

    with c2:

        with st.container(border=True):

            st.markdown("**Contract Type vs Churn**")

            fig, ax = plt.subplots(
                figsize=(4.2, 2.5)
            )

            sns.countplot(
                data=df,
                x="Contract",
                hue="Churn",
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("Customers")
            ax.tick_params(
                axis="x",
                rotation=8,
                labelsize=8
            )

            ax.tick_params(
                axis="y",
                labelsize=8
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 3
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("**Tenure vs Churn**")

            fig, ax = plt.subplots(
                figsize=(4.2, 2.5)
            )

            sns.boxplot(
                data=df,
                x="Churn",
                y="tenure",
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("Months")
            ax.tick_params(labelsize=8)

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 4
    # -----------------------------------------------------

    with c2:

        with st.container(border=True):

            st.markdown("**Monthly Charges vs Churn**")

            fig, ax = plt.subplots(
                figsize=(4.2, 2.5)
            )

            sns.boxplot(
                data=df,
                x="Churn",
                y="MonthlyCharges",
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("Charges")
            ax.tick_params(labelsize=8)

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # GRAPH 5
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("**Internet Service vs Churn**")

            fig, ax = plt.subplots(
                figsize=(4.2, 2.5)
            )

            sns.countplot(
                data=df,
                x="InternetService",
                hue="Churn",
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("Customers")
            ax.tick_params(
                axis="x",
                rotation=8,
                labelsize=8
            )

            ax.tick_params(
                axis="y",
                labelsize=8
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


# =========================================================
# MODEL ANALYSIS
# =========================================================

elif st.session_state.page == "Model":

    st.title("🤖 Model Performance")

    st.caption(
        "Compare the trained machine learning models."
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

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

    st.subheader("Model Comparison")

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

    # -----------------------------------------------------
    # MODEL GRAPH
    # -----------------------------------------------------

    left, center, right = st.columns([1, 2, 1])

    with center:

        with st.container(border=True):

            fig, ax = plt.subplots(
                figsize=(5.0, 2.8)
            )

            x = np.arange(
                len(results)
            )

            width = 0.15

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
                fontsize=8
            )

            ax.set_ylabel(
                "%",
                fontsize=8
            )

            ax.set_ylim(
                0,
                100
            )

            ax.set_title(
                "Model Performance",
                fontsize=11,
                fontweight="bold"
            )

            ax.legend(
                fontsize=6,
                ncol=2
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # TEST DATA
    # -----------------------------------------------------

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

    (
        X_train_analysis,
        X_test_analysis,
        y_train_analysis,
        y_test_analysis
    ) = train_test_split(
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

    predictions = model.predict(
        X_test_analysis_scaled
    )

    probabilities = model.predict_proba(
        X_test_analysis_scaled
    )[:, 1]

    # -----------------------------------------------------
    # CONFUSION MATRIX + ROC
    # -----------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("**Confusion Matrix**")

            cm = confusion_matrix(
                y_test_analysis,
                predictions
            )

            fig, ax = plt.subplots(
                figsize=(3.1, 2.6)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax
            )

            ax.set_xlabel(
                "Predicted",
                fontsize=8
            )

            ax.set_ylabel(
                "Actual",
                fontsize=8
            )

            ax.set_xticklabels(
                ["No Churn", "Churn"],
                fontsize=7
            )

            ax.set_yticklabels(
                ["No Churn", "Churn"],
                fontsize=7,
                rotation=0
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    with c2:

        with st.container(border=True):

            st.markdown("**ROC Curve**")

            fpr, tpr, _ = roc_curve(
                y_test_analysis,
                probabilities
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            fig, ax = plt.subplots(
                figsize=(3.1, 2.6)
            )

            ax.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.2f}"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel(
                "False Positive Rate",
                fontsize=8
            )

            ax.set_ylabel(
                "True Positive Rate",
                fontsize=8
            )

            ax.tick_params(
                labelsize=7
            )

            ax.legend(
                fontsize=8
            )

            ax.set_title(
                "ROC Curve",
                fontsize=11,
                fontweight="bold"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)

    # -----------------------------------------------------
    # FINAL MODEL
    # -----------------------------------------------------

    st.subheader("🏆 Selected Model")

    st.success(
        "Logistic Regression was selected as the final model "
        "because it achieved the strongest overall performance."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            "80.70%"
        )

    with c2:
        st.metric(
            "Precision",
            "65.84%"
        )

    with c3:
        st.metric(
            "Recall",
            "56.68%"
        )

    with c4:
        st.metric(
            "ROC-AUC",
            "84.16%"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ • Customer Churn Prediction System • "
    "Machine Learning Project"
)
