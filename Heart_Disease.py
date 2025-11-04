# AI for SDG 3: Predicting Heart Disease Risk

# Description: Simple ML model to predict heart disease using logistic regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/krishnaik06/Heart-Disease-UCI-Dataset/master/heart.csv"
data = pd.read_csv(url)
print("✅ Dataset loaded successfully!\n")

# Step 2: Inspect the data
print("Dataset Preview:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# Step 3: Prepare features (X) and target (y)
# Note: The dataset uses 'target' as the outcome variable
X = data.drop('target', axis=1)
y = data['target']

# Step 4: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Step 8: Visualize results
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Heart Disease Prediction Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Example prediction (optional)
# Replace these numbers with your own patient data if you want to test
sample = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
result = model.predict(sample)
print("\n🩺 Sample Prediction (1 = heart disease, 0 = healthy):", result)
