import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Streamlit app config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection App")

# Upload dataset
uploaded_file = st.file_uploader("📎 Upload your creditcard.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Balance the data
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    x = new_dataset.drop(columns="Class")
    y = new_dataset["Class"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    # Custom colors
    colors = {
        "Logistic Regression": "Blues",
        "Random Forest": "Greens",
        "XGBoost": "Oranges",
    }

    # Evaluate each model
    for name, model in models.items():
        st.subheader(f"🔍 {name}")
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap=colors[name],
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
            ax=ax1,
        )
        ax1.set_title(f"{name} - Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)

        # ROC Curve
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.2f}",
            color=colors[name].replace("s", "").lower(),
        )
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_title(f"{name} - ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

        # Classification Report
        report_dict = classification_report(
            y_test, y_pred, target_names=["Not Fraud", "Fraud"], output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.dataframe(report_df.style.format(precision=2))

else:
    st.info("Please upload your `creditcard.csv` file to get started.")
