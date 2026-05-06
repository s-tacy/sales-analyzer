# 1. Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


st.title("📊 Sales Data Analyzer Dashboard")
st.caption("An interactive machine learning dashboard for revenue prediction")

st.sidebar.title("Sales Analyzer")
st.sidebar.write("""
This dashboard analyzes sales data and predicts whether a transaction
will generate high revenue using a Logistic Regression model.

Built with:
- pandas
- scikit-learn
- matplotlib
- Streamlit
""")

# 2. Loading data
df = pd.read_csv("data/sales.csv")

# 3. Feature Engineering
df["Revenue"] = df["Price"] * df["Quantity"]
df["High_Revenue"] = df["Revenue"].apply(lambda x: 1 if x > 2000 else 0)

# Encoding
df_encoded = pd.get_dummies(df, columns=["Product", "Region"])

# 4. Model Preparation
X = df_encoded.drop(columns=["Revenue", "High_Revenue"])
y = df_encoded["High_Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🤖 Model", "🧠 Insights"])

# ================== OVERVIEW ==================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df.groupby("Product")["Revenue"].sum().plot(kind="bar", ax=ax)
        ax.set_title("Revenue By Product")
        ax.set_xlabel("Product")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        df.groupby("Region")["Revenue"].sum().plot(
            kind="pie", autopct="%1.1f%%", ax=ax2
        )
        ax2.set_title("Revenue Distribution By Region")
        ax2.set_ylabel("")
        st.pyplot(fig2)

# ================== MODEL ==================
with tab2:
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy)

    st.write("Confusion Matrix:")
    st.dataframe(cm)

    st.markdown("---")
    st.subheader("Make a Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price", min_value=0.0, step=1.0, value=0.0)
        quantity = st.number_input("Quantity", min_value=1, step=1, value=1)

    with col2:
        product = st.selectbox("Product", df["Product"].unique())
        region = st.selectbox("Region", df["Region"].unique())

    use_scaling = st.checkbox("Use Feature Scaling", value=True)

    # Prepare input
    input_data = pd.DataFrame({
        "Price": [price],
        "Quantity": [quantity],
        "Product": [product],
        "Region": [region]
    })

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    if use_scaling:
        input_final = scaler.transform(input_encoded)
    else:
        input_final = input_encoded

    # Prediction
    if st.button("Predict Revenue Class"):

        if price == 0:
            st.warning("Please enter a valid price.")
        else:
            prediction = model.predict(input_final)[0]
            probability = model.predict_proba(input_final)[0]

            if prediction == 1:
                st.success("This transaction is likely to generate high revenue.")
            else:
                st.warning("This transaction may not generate high revenue.")

            st.subheader("📊 Prediction Confidence")

            confidence_df = pd.DataFrame({
                "Class": ["Low Revenue", "High Revenue"],
                "Probability": probability
            })

            st.dataframe(confidence_df)

            # Explanation
            st.subheader("🧠 Prediction Explanation")

            contributions = model.coef_[0] * input_encoded.iloc[0]

            explanation = pd.DataFrame({
                "Feature": X.columns,
                "Impact": contributions
            }).sort_values(by="Impact", ascending=False)

            st.dataframe(explanation)

            top_feature = explanation.iloc[0]
            st.write(
                f"The prediction was mostly influenced by **{top_feature['Feature']}**, "
                f"which had an impact of {top_feature['Impact']:.2f}."
            )

# ================== INSIGHTS ==================
with tab3:
    st.subheader("Feature Importance")
    st.dataframe(feature_importance)
