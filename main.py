import pandas as pd

# Load the dataset
df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    names=["ID", "Topic", "Sentiment", "Text"]
)

print("Shape of Dataset:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

print("\nTop 10 Topics:")
print(df["Topic"].value_counts().head(10))

# ==========================
# Data Cleaning
# ==========================

df = df.drop_duplicates()
df["Text"] = df["Text"].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDataset Shape After Cleaning:")
print(df.shape)

# ==========================
# Build & Evaluate Model (CLEAN VERSION)
# ==========================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("\nPreparing data...")

X = df["Text"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Creating model...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=500))
])

print("Training model... Please wait.")

model.fit(X_train, y_train)

print("Training completed!")

print("Making predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Model Evaluation")
print("==============================")
print(f"Accuracy: {accuracy*100:.2f}%")

# ==========================
# Save Model
# ==========================

import joblib

joblib.dump(model, "sentiment_model.pkl")

print("\nModel saved successfully as sentiment_model.pkl")

# ==========================
# Test Predictions
# ==========================

sample_tweets = [
    "I love this game. It is amazing!",
    "The service is terrible. I hate it.",
    "The update is okay, nothing special."
]

predictions = model.predict(sample_tweets)

print("\nSample Predictions:")

for tweet, sentiment in zip(sample_tweets, predictions):
    print(f"Tweet: {tweet}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 50)

# ==========================
# Visualizations
# ==========================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
df["Sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()