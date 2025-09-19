# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Check first 5 rows
print("First 5 rows of the dataset:\n", data.head())

# Basic info
print("\nDataset Info:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Step 2: Data Visualization

# Count of each crop
plt.figure(figsize=(12,6))
sns.countplot(x='label', data=data)
plt.xticks(rotation=45)
plt.title("Number of Samples for Each Crop")
plt.show()

# 4b: Correlation heatmap (exclude label column)
plt.figure(figsize=(10,6))
sns.heatmap(data.drop(columns='label').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Features")
plt.show()
# Step 3: Prepare Features and Labels
X = data.drop(columns='label')  # Features
y = data['label']               # Target variable

# Step 4: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(model, "crop_model.pkl")
print("\nModel saved as crop_model.pkl")

# Step 7 (Optional): Test Model on New Data
# Example: [N, P, K, temperature, humidity, ph, rainfall]
sample = np.array([[90, 42, 43, 20.8, 82, 6.5, 202]])
predicted_crop = model.predict(sample)
print("\nRecommended crop for sample data:", predicted_crop[0])
