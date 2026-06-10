import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("water.csv")

# Convert complaint text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["complaint_text"])

# Show shape of matrix
print("Matrix shape:", X.shape)

# Show important words
print("\nFeatures:")
print(vectorizer.get_feature_names_out()[:20])