import pandas as pd
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("water.csv")

# Function to clean complaint text
def clean_text(text):
    doc = nlp(text.lower())

    cleaned_words = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            cleaned_words.append(token.lemma_)

    return " ".join(cleaned_words)

# Apply cleaning
df["cleaned_text"] = df["complaint_text"].apply(clean_text)

# Show result
print(df[["complaint_text", "cleaned_text"]].head())