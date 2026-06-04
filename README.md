Citizen Grievance AI - Water Department

Project Overview

Citizen Grievance AI is an NLP-based complaint classification system developed for the Water Department. The system analyzes water-related citizen complaints and predicts the sentiment category of issues using Machine Learning and Natural Language Processing (NLP).

The model classifies complaints into:

- Critical
- Negative
- Neutral
- Positive

Example

Input:

Leakage in water pipeline

Output:

Critical

---

Features

✔ NLP-based complaint sentiment analysis
✔ TF-IDF text vectorization
✔ Logistic Regression classification model
✔ Complaint sentiment prediction system
✔ Exploratory Data Analysis (EDA)
✔ Data visualization graphs
✔ FastAPI integration for real-time prediction
✔ Swagger API documentation ("/docs")

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- FastAPI
- Uvicorn
- VS Code
- Git & GitHub

---

Dataset

The dataset contains water-related citizen complaints categorized into sentiment labels:

- Critical
- Negative
- Neutral
- Positive

Dataset File:
"water.csv"

---

Project Structure

citizen-grievance-ai
│── water.csv
│── water_eda.ipynb
│── train_model.py
│── predict.py
│── api.py
│── vectorize_text.py
│── README.md

---

Model Used

TF-IDF Vectorization

Used to convert complaint text into numerical form for machine learning.

Logistic Regression

Used as the classification algorithm for sentiment prediction.

---

Model Accuracy

Achieved approximately:

80% accuracy

on the Water grievance dataset.

---

How to Run the Project

Install Dependencies

pip install pandas scikit-learn matplotlib jupyter notebook fastapi uvicorn

Run Prediction

python predict.py

Run FastAPI

uvicorn api:app --reload

---

API Documentation

Open browser:

http://127.0.0.1:8000/docs

Swagger UI allows testing complaint predictions directly through the browser.

Sample API Request

{
  "complaint": "Leakage in water pipeline"
}

Sample API Response

{
  "complaint": "Leakage in water pipeline",
  "predicted_sentiment": "Critical"
}

---

Author

[pooja gupta]