import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("water.csv")

# Remove empty values (NaN)
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

# Convert text to numbers
X_vectorized = vectorizer.fit_transform(X)

# Improved model
model = LogisticRegression(
    max_iter=3000,
    C=5,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_vectorized, y)

# Take complaint from user
user_input = input("Enter complaint: ")

# Convert complaint text
user_vector = vectorizer.transform([user_input])

# Predict sentiment
prediction = model.predict(user_vector)

# Output
print("Predicted Sentiment:", prediction[0])