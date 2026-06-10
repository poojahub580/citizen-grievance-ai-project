import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Improved model
model = LogisticRegression(
    max_iter=3000,
    C=5,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model and vectorizer
joblib.dump(model, "grievance_model.pkl")
joblib.dump(vectorizer, "text_encoder.pkl")

print("Model saved successfully.")
print("Vectorizer saved successfully.")