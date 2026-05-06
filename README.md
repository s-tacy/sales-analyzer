# 📊 Sales Data Analysis & Machine Learning Dashboard

## 🧠 Overview

This project analyzes sales transaction data to uncover key business insights and uses machine learning to predict whether a transaction will result in high or low revenue.

It demonstrates an **end-to-end data science workflow**, extended into an **interactive Streamlit dashboard** where users can explore data and generate real-time predictions.

The project covers:

* Data cleaning and preprocessing
* Feature engineering
* Data visualization
* Machine learning (classification)
* Model evaluation and interpretation
* Interactive dashboard deployment

---

## 🎯 Objective

The goal of this project is to:

* Understand what drives high-revenue sales
* Analyze product and regional performance
* Build a machine learning model to classify transactions as:

  * High Revenue (1)
  * Low Revenue (0)
* Provide an **interactive interface** for users to test predictions

---

## 🚀 Key Features

* 📊 **Data Visualization**

  * Revenue by Product
  * Revenue by Region
  * Revenue distribution

* 🤖 **Machine Learning Model**

  * Logistic Regression classifier
  * Predicts High vs Low revenue transactions

* 🎛️ **Interactive Dashboard (Streamlit)**

  * User inputs (Price, Quantity, Product, Region)
  * Real-time predictions
  * Optional feature scaling toggle

* 🧠 **Model Explainability**

  * Feature contribution breakdown
  * Highlights the most influential factors in predictions

---

## 📦 Dataset Description

The dataset contains sales transaction-level data with the following features:

* **Product**: Type of product sold
* **Price**: Unit price of the product
* **Quantity**: Number of units sold
* **Region**: Sales region
* **Revenue**: Calculated as `Price × Quantity`

---

## 🔄 Project Workflow

### 1. Data Preparation

* Loaded dataset using pandas
* Checked structure and consistency
* Created new feature: `Revenue`

### 2. Feature Engineering

* Created target variable:

  * High Revenue = 1
  * Low Revenue = 0
* Applied one-hot encoding to categorical variables

### 3. Data Splitting

* Split dataset into training (80%) and testing (20%) sets

### 4. Feature Scaling

* Applied StandardScaler to normalize numerical features

### 5. Model Training

* Trained a Logistic Regression classifier

### 6. Prediction & Evaluation

* Evaluated using accuracy score
* Analyzed performance using confusion matrix

### 7. Deployment (Streamlit)

* Built an interactive dashboard
* Enabled real-time user predictions
* Integrated model explainability

---

## 🤖 Machine Learning Model

### Model Used:

* Logistic Regression

### Why this model?

* Suitable for binary classification
* Interpretable results
* Efficient for small to medium datasets

---

## 📊 Model Performance

* **Accuracy:** 85.7%

### Confusion Matrix:

```
[[2 1]
 [0 4]]
```

### Interpretation:

* The model correctly identifies most high-revenue transactions
* No high-revenue cases were missed (0 false negatives)
* Minor misclassification of low-revenue transactions

---

## 🧠 Feature Importance Insights

The model revealed that:

* **Quantity sold** strongly influences revenue classification
* **Product type** significantly affects revenue outcome
* **Region** has moderate influence on sales performance

---

## 📈 Visualizations

The project includes the following visual analyses:

* Revenue by Product
* Revenue by Region
* Revenue Distribution
* Confusion Matrix
* Feature Importance

*(Optional: add images if available in `/outputs/` folder)*

---

## 🌐 Interactive Dashboard

The project includes a **Streamlit dashboard** that allows users to:

* Explore the dataset visually
* View model performance metrics
* Input custom transaction details
* Receive real-time predictions
* Understand *why* a prediction was made

---

## 📂 Project Structure

```id="struct1"
Sales Data Analyzer/
│
├── app/
│   └── app.py              # Streamlit dashboard
│
├── src/
│   └── sales_analyzer.py   # Data processing & ML pipeline
│
├── data/
│   └── sales.csv           # Dataset
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/sales-data-analyzer.git
cd sales-data-analyzer
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run app/app.py
```

---

## 💡 Key Insights

* High revenue is primarily driven by **product type and quantity sold**
* Certain products consistently generate higher revenue
* Sales distribution shows variation between low and high-value transactions
* The model performs well in identifying high-value sales opportunities

---

## 🚀 Future Improvements

* Experiment with advanced models (Decision Trees, Random Forest, XGBoost)
* Add time-based sales trends (time series analysis)
* Improve dataset size and realism
* Deploy dashboard publicly (Streamlit Cloud)
* Perform hyperparameter tuning

---

## 🛠️ Tech Stack

* Python
* pandas
* matplotlib
* scikit-learn
* Streamlit

---

## 🧡 What I Learned

* Building an end-to-end ML pipeline
* Feature engineering and encoding strategies
* Model evaluation and interpretation
* Turning ML models into interactive applications
* Structuring a project for real-world use

---

## 👤 Author

**Stacy Mwaura**
Computer Science Student | Aspiring Data Scientist

---

## 📌 Summary

This project demonstrates how raw sales data can be transformed into actionable insights, predictive models, and an interactive user-facing application using Python and machine learning.

