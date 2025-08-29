# churn_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
print("📂 Loading dataset...")
df = pd.read_csv("churn.csv")
print("✅ Data loaded successfully")
print("Shape of dataset:", df.shape)

# 2. Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Split into features and target
X = df.drop("Churn_Yes", axis=1)  # target is "Churn"
y = df["Churn_Yes"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into training and testing sets")

# 5. Scale the features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train Logistic Regression model
model = LogisticRegression(max_iter=5000)  # increased iterations
model.fit(X_train, y_train)
print("✅ Model training complete")

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Optional: plot confusion matrix
import seaborn as sns

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


