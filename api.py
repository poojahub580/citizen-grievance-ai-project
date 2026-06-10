from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load saved model and text encoder
model = joblib.load("grievance_model.pkl")
vectorizer = joblib.load("text_encoder.pkl")


# Input structure
class Complaint(BaseModel):
    complaint: str


# Prediction API
@app.post("/predict")
def predict(data: Complaint):

    # Convert complaint text into vector
    text = vectorizer.transform([data.complaint])

    # Predict sentiment
    prediction = model.predict(text)

    return {
        "complaint": data.complaint,
        "predicted_sentiment": prediction[0]
    }


# Home route
@app.get("/")
def home():
    return {
        "message": "Water Grievance AI API is running successfully"
    }