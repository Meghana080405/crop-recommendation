# 🌱 Crop Recommendation System using Machine Learning

## 📌 Project Overview
This project predicts the best crop to cultivate based on soil and environmental conditions using **Machine Learning**.  
It is trained on a dataset of 2200 samples containing values of N, P, K (soil nutrients), temperature, humidity, pH, and rainfall.

## 🛠️ Technologies Used
- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn (Random Forest Classifier)
- Joblib (for saving ML model)

## 🚀 Features
- Takes soil nutrients and weather inputs from the user.
- Predicts the most suitable crop with **99% accuracy**.
- Easy to run on command line with Python.

## 📂 Project Structure
crop-ml-project/
│-- Crop_recommendation.csv # Dataset
│-- crop_analysis.py # Data exploration + model training
│-- crop_recommendation_app.py # User input & prediction
│-- crop_model.pkl # Trained model
│-- venv/ # Virtual environment (not needed for GitHub)

shell
## 📊 Sample Run
🌱 Welcome to the Crop Recommendation System 🌱
Enter Nitrogen content (N): 80
Enter Phosphorus content (P): 60
Enter Potassium content (K): 40
Enter Temperature (°C): 26
Enter Humidity (%): 70
Enter Soil pH: 6.7
Enter Rainfall (mm): 140

✅ Recommended crop for your field: Jute

markdown
## 📈 Model Performance
- Accuracy: **99%**
- Algorithm used: **Random Forest Classifier**
- Confusion Matrix and Classification Report generated for evaluation.

## 🧑‍💻 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/Meghana080405/crop-recommendation.git
Create a virtual environment:

python -m venv venv
Install dependencies:

pip install -r requirements.txt
Run the application:

python crop_recommendation_app.py
🌾 Output Example
The system suggests crops like:

Rice 🌾

Maize 🌽

Jute 🌱

Mango 🥭

Papaya 🍍

Cotton 🌿

Coffee ☕ ... and more.

⭐ If you like this project, don’t forget to star this repo!










