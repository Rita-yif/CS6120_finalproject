import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

# Load dataset and skip malformed rows
try:
    data = pd.read_csv("stockerbot-export.csv", on_bad_lines='skip')
    print("Data loaded successfully. Number of rows:", len(data))
    print("Dataset columns:", data.columns)
except Exception as e:
    print(f"Failed to load data: {e}")

# Verify required columns
required_columns = ['text', 'timestamp']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
else:
    print("Dataset contains all required columns!")

# Convert timestamp to datetime
try:
    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%a %b %d %H:%M:%S %z %Y")
    print("Timestamp successfully parsed!")
except ValueError:
    print("Unable to parse timestamp format. Attempting automatic detection.")
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    print("Invalid timestamps removed:", data['timestamp'].isna().sum())
    data = data.dropna(subset=['timestamp'])

# Clean text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

data['clean_text'] = data['text'].apply(clean_text)
print("Text cleaning completed.")

# Sample 1000 rows
sampled_data = data.sample(1000, random_state=42)

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.to(device)

# Predict sentiment in batches
def predict_sentiment_in_batches(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = torch.argmax(predictions, axis=1)
        results.extend(labels.cpu().tolist())
    return results

# FinBERT sentiment prediction
print("Starting FinBERT sentiment analysis...")
sampled_data['finbert_label'] = predict_sentiment_in_batches(sampled_data['clean_text'].tolist(), batch_size=32)
print("FinBERT sentiment analysis completed!")

# Save results
sampled_data.to_csv("stockerbot_sentiment_results.csv", index=False)
print("Results saved to 'stockerbot_sentiment_results.csv'")

# Plot sentiment distribution for FinBERT
finbert_sentiment_counts = sampled_data['finbert_label'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(finbert_sentiment_counts.index, finbert_sentiment_counts.values, color='blue')
plt.xlabel("Sentiment Labels (FinBERT)")
plt.ylabel("Frequency")
plt.title("Sentiment Distribution (FinBERT Predictions)")
plt.show()

# Logistic Regression using FinBERT labels
X = sampled_data['clean_text']
y = sampled_data['finbert_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# Logistic Regression predictions
y_pred_logistic = lr.predict(X_test_tfidf)

# Evaluate Logistic Regression performance
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))

# Predict labels for the entire sampled dataset
sampled_data['logistic_label'] = lr.predict(vectorizer.transform(sampled_data['clean_text']))

# Plot sentiment distribution for Logistic Regression
logistic_sentiment_counts = sampled_data['logistic_label'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(logistic_sentiment_counts.index, logistic_sentiment_counts.values, color='green')
plt.xlabel("Sentiment Labels (Logistic Regression)")
plt.ylabel("Frequency")
plt.title("Sentiment Distribution (Logistic Regression Predictions)")
plt.show()

# Compare consistency between FinBERT and Logistic Regression
accuracy = accuracy_score(sampled_data['finbert_label'], sampled_data['logistic_label'])
print("Consistency between FinBERT and Logistic Regression:", accuracy)
print("Classification Report (FinBERT vs Logistic Regression):")
print(classification_report(sampled_data['finbert_label'], sampled_data['logistic_label']))

# Filter negative sentiment tweets (label = 2)
negative_tweets = sampled_data[sampled_data['finbert_label'] == 2]

if not negative_tweets.empty:
    print(f"Number of negative sentiment tweets: {len(negative_tweets)}")
    print(negative_tweets[['text', 'finbert_label']].head())
else:
    print("No negative sentiment tweets detected.")


# Group by date and count negative tweets
negative_trend = negative_tweets.groupby(negative_tweets['timestamp'].dt.date).size()

# Plot negative sentiment trend
plt.figure(figsize=(12, 6))
plt.plot(negative_trend.index, negative_trend.values, marker='o', linestyle='-', color='red')
plt.xlabel('Date')
plt.ylabel('Negative Sentiment Count')
plt.title('Negative Sentiment Trend Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.show()


from sklearn.feature_extraction.text import CountVectorizer

# Extract top keywords from negative sentiment tweets
vectorizer = CountVectorizer(stop_words='english', max_features=20)
negative_words = vectorizer.fit_transform(negative_tweets['clean_text'])
word_counts = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'count': negative_words.toarray().sum(axis=0)
}).sort_values(by='count', ascending=False)

# Print top keywords
print("Top keywords in negative sentiment tweets:\n", word_counts)

# Plot top keywords
plt.figure(figsize=(10, 6))
plt.bar(word_counts['word'], word_counts['count'], color='red')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top Negative Sentiment Words")
plt.xticks(rotation=45)
plt.show()


if 'company_names' in negative_tweets.columns:
    company_counts = negative_tweets['company_names'].value_counts()
    print("Negative sentiment distribution by company:\n", company_counts)

    # Plot company distribution
    plt.figure(figsize=(12, 6))
    company_counts[:10].plot(kind='bar', color='red')
    plt.xlabel("Company")
    plt.ylabel("Negative Sentiment Count")
    plt.title("Negative Sentiment Distribution by Company")
    plt.show()


# Detect anomalies in negative sentiment trend
threshold = negative_trend.mean() * 2  # Define anomaly threshold
anomalies = negative_trend[negative_trend > threshold]
print("Detected anomaly dates:\n", anomalies)
