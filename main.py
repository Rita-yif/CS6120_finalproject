import pandas as pd
import re
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

# CHECK GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

# DATASET
data = pd.read_csv("stockerbot-export.csv", on_bad_lines='skip')
print("Data loaded. Number of rows:", len(data))



# DATA REPROCESS
required_columns = ['text', 'timestamp']


try:

    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    print("Timestamp successfully parsed with specified format!")
except Exception as e:
    print("Failed to parse timestamp with specified format:", e)

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    print("Invalid timestamps removed:", data['timestamp'].isna().sum())
    data = data.dropna(subset=['timestamp'])
    print("Timestamp column cleaned successfully.")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # 移除 URL
    text = re.sub(r"[^\w\s]", "", text)  # 移除标点符号
    return text

data['clean_text'] = data['text'].apply(clean_text)

# sample
sampled_data = data.sample(1000, random_state=42)

#  FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.to(device)




# predict dentiment
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


print("Starting sentiment analysis...")
sampled_data['finbert_label'] = predict_sentiment_in_batches(sampled_data['clean_text'].tolist(), batch_size=32)
print("Sentiment analysis completed!")


sampled_data.to_csv("sentiment_results.csv", index=False)


# Save FinBERT results
sampled_data.to_csv("sentiment_results.csv", index=False)

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

# Stratified train-test split
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in stratified_split.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)

# Logistic Regression predictions and evaluation
y_pred_logistic = lr.predict(X_test_tfidf)
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
# 绘制正面与负面情感时间趋势
positive_tweets = sampled_data[sampled_data['finbert_label'] == 0]
negative_tweets = sampled_data[sampled_data['finbert_label'] == 2]

positive_trend = positive_tweets.groupby(positive_tweets['timestamp'].dt.date).size()
negative_trend = negative_tweets.groupby(negative_tweets['timestamp'].dt.date).size()

plt.figure(figsize=(12, 6))
plt.plot(positive_trend.index, positive_trend.values, marker='o', linestyle='-', label='Positive Sentiment', color='green')
plt.plot(negative_trend.index, negative_trend.values, marker='o', linestyle='-', label='Negative Sentiment', color='red')
plt.xlabel('Date')
plt.ylabel('Tweet Count')
plt.title('Positive and Negative Sentiment Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# extract positive and negative high frequency words
def extract_top_keywords(tweets, label, top_n=20):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    text_data = vectorizer.fit_transform(tweets['clean_text'])
    word_counts = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'count': text_data.toarray().sum(axis=0)
    }).sort_values(by='count', ascending=False)
    print(f"Top {top_n} keywords for label {label}:\n", word_counts)
    return word_counts

# positive word
if not positive_tweets.empty:
    positive_keywords = extract_top_keywords(positive_tweets, label=0)
    plt.figure(figsize=(10, 6))
    plt.bar(positive_keywords['word'], positive_keywords['count'], color='green')
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top Positive Sentiment Keywords")
    plt.xticks(rotation=45)
    plt.show()

# negative word
if not negative_tweets.empty:
    negative_keywords = extract_top_keywords(negative_tweets, label=2)
    plt.figure(figsize=(10, 6))
    plt.bar(negative_keywords['word'], negative_keywords['count'], color='red')
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top Negative Sentiment Keywords")
    plt.xticks(rotation=45)
    plt.show()

# Pie chart for sentiment distribution (FinBERT)
finbert_sentiment_proportions = sampled_data['finbert_label'].value_counts(normalize=True)
plt.figure(figsize=(8, 8))
plt.pie(finbert_sentiment_proportions, labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', colors=['green', 'blue', 'red'])
plt.title('Sentiment Proportions (FinBERT)')
plt.show()

# Pie chart for sentiment distribution (Logistic Regression)
logistic_sentiment_proportions = sampled_data['logistic_label'].value_counts(normalize=True)
plt.figure(figsize=(8, 8))
plt.pie(logistic_sentiment_proportions, labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', colors=['green', 'blue', 'red'])
plt.title('Sentiment Proportions (Logistic Regression)')
plt.show()

# Positive and Negative Sentiment Trends (Comparison)
positive_trend_lr = positive_tweets.groupby(positive_tweets['timestamp'].dt.date)['logistic_label'].count()
negative_trend_lr = negative_tweets.groupby(negative_tweets['timestamp'].dt.date)['logistic_label'].count()

plt.figure(figsize=(14, 8))
plt.plot(positive_trend.index, positive_trend.values, marker='o', linestyle='-', label='Positive Sentiment (FinBERT)', color='green')
plt.plot(negative_trend.index, negative_trend.values, marker='o', linestyle='-', label='Negative Sentiment (FinBERT)', color='red')
plt.plot(positive_trend_lr.index, positive_trend_lr.values, marker='x', linestyle='--', label='Positive Sentiment (Logistic Regression)', color='lightgreen')
plt.plot(negative_trend_lr.index, negative_trend_lr.values, marker='x', linestyle='--', label='Negative Sentiment (Logistic Regression)', color='orange')
plt.xlabel('Date')
plt.ylabel('Tweet Count')
plt.title('Sentiment Trends Over Time (Comparison)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

from wordcloud import WordCloud

# Word Cloud for Positive Sentiment
if not positive_tweets.empty:
    positive_text = " ".join(positive_tweets['clean_text'])
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Sentiment')
    plt.show()

# Word Cloud for Negative Sentiment
if not negative_tweets.empty:
    negative_text = " ".join(negative_tweets['clean_text'])
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Sentiment')
    plt.show()

import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Generate co-occurrence matrix for positive sentiment
def plot_cooccurrence_matrix(tweets, title):
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform(tweets['clean_text'])
    cooccurrence_matrix = (X.T * X).toarray()
    words = vectorizer.get_feature_names_out()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence_matrix, annot=False, cmap='Greens', xticklabels=words, yticklabels=words)
    plt.title(title)
    plt.show()

# Positive Sentiment Co-occurrence
if not positive_tweets.empty:
    plot_cooccurrence_matrix(positive_tweets, "Positive Sentiment Keyword Co-occurrence")

# Negative Sentiment Co-occurrence
if not negative_tweets.empty:
    plot_cooccurrence_matrix(negative_tweets, "Negative Sentiment Keyword Co-occurrence")

# Model Performance Comparison
labels = ['Positive', 'Neutral', 'Negative']
finbert_metrics = [0.97, 0.89, 0.81]  # Replace with actual precision values from FinBERT
logistic_metrics = [0.85, 0.52, 0.45]  # Replace with actual precision values from Logistic Regression

x = range(len(labels))
plt.figure(figsize=(10, 6))
plt.bar(x, finbert_metrics, width=0.4, label='FinBERT', align='center', color='blue')
plt.bar([p + 0.4 for p in x], logistic_metrics, width=0.4, label='Logistic Regression', align='center', color='green')
plt.xlabel('Sentiment Labels')
plt.ylabel('Precision')
plt.title('Model Performance Comparison by Precision')
plt.xticks([p + 0.2 for p in x], labels)
plt.legend()
plt.show()

# Highlight anomalies
threshold = negative_trend.mean() * 2  # Threshold for anomalies
anomalies = negative_trend[negative_trend > threshold]

plt.figure(figsize=(12, 6))
plt.plot(negative_trend.index, negative_trend.values, marker='o', label='Negative Sentiment', color='red')
plt.scatter(anomalies.index, anomalies.values, color='black', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Negative Sentiment Count')
plt.title('Anomalies in Negative Sentiment Trend')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

