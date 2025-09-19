import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_model.pkl")

# Function to predict crop
def predict_crop():
    try:
        N = float(entry_N.get())
        P = float(entry_P.get())
        K = float(entry_K.get())
        temperature = float(entry_temperature.get())
        humidity = float(entry_humidity.get())
        ph = float(entry_ph.get())
        rainfall = float(entry_rainfall.get())

        # Create input array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        messagebox.showinfo("Recommended Crop", f"🌱 Recommended crop: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# Tkinter window
root = tk.Tk()
root.title("Crop Recommendation System")
root.geometry("400x450")
root.resizable(False, False)

tk.Label(root, text="🌾 Crop Recommendation System 🌾", font=("Helvetica", 16, "bold")).pack(pady=10)

# Input fields
fields = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature (°C)", 
          "Humidity (%)", "Soil pH", "Rainfall (mm)"]
entries = []

for field in fields:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=field+":", width=15, anchor='w').pack(side='left')
    entry = tk.Entry(frame, width=20)
    entry.pack(side='left')
    entries.append(entry)

entry_N, entry_P, entry_K, entry_temperature, entry_humidity, entry_ph, entry_rainfall = entries

# Predict button
tk.Button(root, text="Predict Crop", command=predict_crop, bg="green", fg="white").pack(pady=20)

root.mainloop()
