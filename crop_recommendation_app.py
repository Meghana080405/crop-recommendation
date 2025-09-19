import pandas as pd
import joblib

# Load the trained model
model = joblib.load("crop_model.pkl")

print("🌱 Welcome to the Crop Recommendation System 🌱\n")

# Get user input
N = int(input("Enter Nitrogen content (N): "))
P = int(input("Enter Phosphorus content (P): "))
K = int(input("Enter Potassium content (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Create DataFrame with proper column names
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# Make prediction
recommended_crop = model.predict(input_data)[0]

print(f"\n✅ Recommended crop for your field: {recommended_crop}")
