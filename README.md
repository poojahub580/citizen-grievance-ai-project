Citizen Grievance AI – Water Department

Project Overview

Citizen Grievance AI is a Natural Language Processing (NLP) and Machine Learning based complaint classification system developed for the Water Department. The system analyzes water-related citizen grievances and predicts complaint sentiment categories to help authorities identify issue severity and improve grievance prioritization.

The project uses TF-IDF Vectorization and Logistic Regression to classify complaints into the following sentiment categories:

- Critical
- Negative
- Neutral
- Positive

Example

Input:

Leakage in water pipeline near residential area

Output:

Critical

---

Features

✔ NLP-based complaint sentiment analysis
✔ TF-IDF text vectorization
✔ Logistic Regression classification model
✔ Water complaint sentiment prediction system
✔ Exploratory Data Analysis (EDA)
✔ Data visualization graphs and charts
✔ Confusion Matrix for model evaluation
✔ FastAPI integration for real-time prediction
✔ Swagger API documentation ("/docs")
✔ Saved model and vectorizer using Joblib

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- FastAPI
- Uvicorn
- Joblib
- VS Code
- Git & GitHub

---

Dataset

The dataset contains water-related citizen grievances categorized into sentiment labels:

- Critical
- Negative
- Neutral
- Positive

Dataset File

water.csv

The dataset includes complaints such as:

- Water pipeline leakage
- Dirty water supply
- No water supply
- Low water pressure
- Water contamination issues
- Water supply restoration feedback

---

Project Structure

citizen-grievance-ai
│── water.csv
│── water_eda.ipynb
│── complaint_classifier.py
│── api.py
│── complaint_cleaner.py
│── grievance_loader.py
│── grievance_predictor.py
│── text_feature_builder.py
│── grievance_model.pkl
│── text_encoder.pkl
│── README.md
│── requirements.txt

---

Model Used

TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts complaint text into numerical vectors that help the machine learning model understand important complaint-related words.

Logistic Regression

Logistic Regression is used as the classification algorithm for predicting sentiment categories. It was selected due to its effectiveness in text classification and NLP tasks.

---

Model Accuracy

The trained machine learning model achieved approximately:

80% Accuracy

on the Water Grievance Dataset.

---

Exploratory Data Analysis (EDA)

EDA was performed to better understand complaint patterns and dataset distribution.

The notebook includes:

- Sentiment Distribution Graph
- Pie Chart Visualization
- Word Frequency Analysis
- Word Cloud Visualization
- Classification Metrics Graph
- Confusion Matrix

These visualizations helped evaluate dataset quality and model performance.

---

How to Run the Project

1. Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn jupyter fastapi uvicorn joblib pydantic

---

2. Train the Model

Run:

python complaint_classifier.py

This generates:

grievance_model.pkl
text_encoder.pkl

---

3. Run FastAPI Server

Run:

uvicorn api:app --reload

---

API Documentation

Open browser:

http://127.0.0.1:8000/docs

Swagger UI allows users to test water grievance predictions directly in the browser.

Sample API Request

{
  "complaint": "Leakage in water pipeline near apartment"
}

Sample API Response

{
  "complaint": "Leakage in water pipeline near apartment",
  "predicted_sentiment": "Critical"
}

---

Future Scope

This project can be extended into a multi-department grievance system, including:

- Roads Department
- Electricity Department
- Smart City Complaint Management

Future improvements may include multilingual complaint support and enhanced machine learning models.

---

Author

Pooja Gupta