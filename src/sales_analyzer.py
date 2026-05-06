#1.imports
import pandas as pd
import matplotlib.pyplot as plt

import os
os.makedirs("outputs", exist_ok=True)

import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#2.load the data
df = pd.read_csv("../data/sales.csv")
print(df.head())

#3.Data Preprocessing
#creating a derived feature
df["Revenue"] = df["Price"] * df["Quantity"]
print(df.head())

#total and average revenue
total_revenue = df["Revenue"].sum()
avg_revenue = df["Revenue"].mean()

#revenue per product and region
product_revenue = df.groupby("Product")["Revenue"].sum()
region_revenue = df.groupby("Region")["Revenue"].sum()

#top products and regions
top_products = product_revenue.sort_values(ascending=False).head(10)
top_regions = region_revenue.sort_values(ascending=False).head(10)

#most profitable product
most_profitable = product_revenue.idxmax()

#best performing region
best_region = region_revenue.idxmax()

#highest single sale
highest_sale = df["Revenue"].max()


#4.Feature Engineering
#ML-predicting if a sale will be High or Low
df["High_Revenue"] = df["Revenue"].apply(lambda x: 1 if x > 2000 else 0)
print(df.head())

#Feature encoding categorical data-numbers
df_encoded = pd.get_dummies(df, columns=["Product","Region"])
print(df_encoded.head())

#5.Train-Test Split
#X features are used by the model to make predictions
X = df_encoded.drop(columns=["Revenue", "High_Revenue"])
#y features are used as the target
y = df_encoded["High_Revenue"]

#Splitting data into training and testing sets
#training-teach the model testing-check if it learned

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

#6.Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#7.Training the model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

#8.Model predictions
y_pred = model.predict(X_test)

#9.Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n Model Accuracy:", accuracy)

#confusion matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix: ", confusion_matrix)


#10.Feature Importance

feature_importance = pd.DataFrame({
    "Feature":X.columns,
    "Coefficient":model.coef_[0]
})

feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)
print("\n Feature Importance:", feature_importance)

#11.Graph visualizations

#a.Product revenue ranking-which product makes the most money?
plt.figure()

product_revenue.sort_values(ascending=False).plot(
    kind="barh",
    title="Product Revenue Ranking")

plt.xlabel("Revenue")
plt.tight_layout
plt.savefig("outputs/product_revenue.png")
plt.show()

#b.Regional revenue-which region drives the business?
plt.figure()

region_revenue.plot(
    kind="pie",
    autopct="%1.2f%%",
    startangle=90,
    title="Regional Revenue Contribution"
)

plt.ylabel("")
plt.savefig("outputs/region_revenue.png")
plt.show()

#c.Transactions distribution-are most sales small, large or medium?
plt.figure()

df["Revenue"].plot(
    kind="hist",
    bins=8,
    color="teal",
    edgecolor="black",
    title="Revenue Distribution per Transaction")

plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig("outputs/revenue_histogram.png")
plt.show()

#d.Confusion Matrix
plt.figure()

sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

#e.Feature importance
plt.figure(figsize=(10,10))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Coefficient"],
)

plt.title("Feature Importance (Logistic Regression)")
plt.xlabel("Impact on High Revenue Prediction")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()


#---------INSIGHTS------------#

print("\n ----Business Sales Insights----")

print("Total Revenue:", total_revenue)
print("Average Revenue per transaction:", avg_revenue)

print("Top Products By Revenue:", top_products)
print("\nTop Regions By Revenue:", top_regions)
print("\nBest Product:", most_profitable)

print("Highest Single Sales:", highest_sale)
print("Best Region:", best_region)















