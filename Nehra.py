# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="HDFC Loan Approval Prediction", layout="wide")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hdfc_loan_dataset.csv")

df = load_data()
st.title("🏦 HDFC Loan Approval Prediction App")

# ---------------------------
# Sidebar - Navigation
# ---------------------------
menu = ["📊 Dataset", "📈 Train Model", "🔮 Predict Loan Status"]
choice = st.sidebar.radio("Navigation", menu)

# ---------------------------
# Encode categorical variables
# ---------------------------
categorical_cols = ["Employment_Status", "Property_Area", "Education", "Married", "Loan_Status"]
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# Pages
# ---------------------------
if choice == "📊 Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### Class Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

elif choice == "📈 Train Model":
    st.subheader("Model Evaluation")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("✅ Accuracy", f"{acc*100:.2f}%")

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected","Approved"],
                yticklabels=["Rejected","Approved"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

elif choice == "🔮 Predict Loan Status":
    st.subheader("Enter Applicant Details")

    # ---------------------------
    # Input fields
    # ---------------------------
    applicant_income = st.number_input("Applicant Income (Monthly)", 10000, 200000, 50000)
    coapplicant_income = st.number_input("Co-applicant Income (Monthly)", 0, 100000, 0)
    loan_amount = st.number_input("Loan Amount", 50000, 10000000, 300000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360])
    credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    employment_status = st.selectbox("Employment Status", le_dict["Employment_Status"].classes_)
    property_area = st.selectbox("Property Area", le_dict["Property_Area"].classes_)
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", le_dict["Education"].classes_)
    married = st.selectbox("Married", le_dict["Married"].classes_)

    # ---------------------------
    # Encode inputs correctly
    # ---------------------------
    input_data = pd.DataFrame([[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,  # already numeric
        le_dict["Employment_Status"].transform([employment_status])[0],
        le_dict["Property_Area"].transform([property_area])[0],
        dependents,
        le_dict["Education"].transform([education])[0],
        le_dict["Married"].transform([married])[0]
    ]], columns=X.columns)

    # Scale
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Loan Approval"):
        prediction = model.predict(input_scaled)[0]
        result = le_dict["Loan_Status"].inverse_transform([prediction])[0]

        if result == "Approved":
            st.success("🎉 Loan Approved!")
        else:
            st.error("❌ Loan Rejected.")
