# Citizen Grievance AI - Water Department

## Project Overview
Citizen Grievance AI is an NLP-based complaint classification system developed for the Water Department. The system analyzes citizen complaints and predicts sentiment categories of water-related issues.

The model classifies complaints into:

- Critical
- Negative
- Neutral
- Positive

Example:

Input:
```text
No water supply in our locality
```

Output:
```text
Critical
```

---

## Features
- NLP-based complaint sentiment analysis
- TF-IDF text vectorization
- Logistic Regression model
- Complaint prediction system
- Exploratory Data Analysis (EDA)
- Confusion Matrix visualization
- Word Cloud generation

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- VS Code

---

## Dataset
The dataset contains water-related citizen complaints categorized into sentiment labels:

- Critical
- Negative
- Neutral
- Positive

Dataset File:
`water.csv`

---

## Project Structure

```text
water-grievance-ai
│── water.csv
│── water_eda.ipynb
│── train_model.py
│── predict.py
│── README.md
```

---

## Model Used

### TF-IDF Vectorization
Used to convert complaint text into numerical form.

### Logistic Regression
Used for complaint sentiment classification.

---

## Model Accuracy
Achieved approximately:

**82.5%**

accuracy on the Water grievance dataset.

---

## How to Run

### Install dependencies
```bash
pip install pandas scikit-learn matplotlib jupyter notebook
```

### Train model
```bash
python train_model.py
```

### Predict complaint sentiment
```bash
python predict.py
```

---

## Sample Prediction

Input:
```text
Water pipeline repaired successfully
```

Output:
```text
Positive
```

---

## Author
(Project Team)