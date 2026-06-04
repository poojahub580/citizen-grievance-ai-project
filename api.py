from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("water.csv")

# Remove empty values
df = df.dropna()

# Features and target
X = df["complaint_text"]
y = df["sentiment"]

# Better text processing
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words="english",
    max_features=3000
)

# Convert text into vectors
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(
    max_iter=3000,
    C=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_vectorized, y)

# Input format
class Complaint(BaseModel):
    complaint: str

# Prediction API
@app.post("/predict")
def predict(data: Complaint):

    text = vectorizer.transform([data.complaint])
    prediction = model.predict(text)

    return {
        "complaint": data.complaint,
        "predicted_sentiment": prediction[0]
    }