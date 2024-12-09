# CS6120_finalproject
# README

## Project Title
**Financial Sentiment Analysis on Stock-Related Tweets Using NLP Models**

---

## Project Overview
This project uses Natural Language Processing (NLP) techniques to analyze stock-related tweets. The primary goal is to evaluate sentiment trends over time, extract insights, and compare the performance of two models: **FinBERT** and **Logistic Regression**. The analysis includes generating sentiment labels, keyword extraction, and visualizations for better understanding market sentiment.

---

## Prerequisites

### Software Requirements:
- Python 3.7 or later
- Libraries:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - transformers
  - torch
  - wordcloud

### Hardware Requirements:
- A machine with GPU support (optional but recommended for faster FinBERT predictions).

---

## Installation Instructions

### Step 1: Install Python Libraries
Run the following command in your terminal to install all required libraries:
```bash
pip install pandas matplotlib seaborn scikit-learn transformers torch wordcloud
```

### Step 2: Clone the Repository
Clone this repository or download the code files.

### Step 3: Prepare the Dataset
- Place the dataset file `stockerbot-export.csv` in the same directory as the script.
- Ensure the dataset contains the following columns:
  - `text` (tweet content)
  - `timestamp` (date and time of the tweet)

---

## Running the Code

### Step 1: Run the Script
To execute the analysis, run the Python script:
```bash
python sentiment_analysis.py
```

### Step 2: View the Outputs
1. **Sentiment Predictions**:
   - A CSV file named `sentiment_results.csv` will be generated, containing:
     - Cleaned tweet text
     - FinBERT sentiment labels
     - Logistic Regression sentiment labels
2. **Visualizations**:
   - Sentiment distribution for both models.
   - Sentiment trends over time.
   - Keyword analysis (positive and negative keywords).
   - Word clouds for sentiment labels.
   - Keyword co-occurrence heatmaps.

3. **Model Performance Metrics**:
   - A classification report comparing FinBERT and Logistic Regression performance will be printed in the terminal.

---

## Key Functions in the Script

### 1. **Data Preprocessing**
- Converts timestamps to a standardized datetime format.
- Cleans tweets by removing URLs, punctuation, and converting text to lowercase.

### 2. **Sentiment Analysis**
- **FinBERT**:
  - Predicts sentiment labels (positive, neutral, negative).
  - Uses the pre-trained model `yiyanghkust/finbert-tone`.
- **Logistic Regression**:
  - Trains a Logistic Regression model on TF-IDF features using FinBERT's labels as ground truth.

### 3. **Insights and Visualizations**
- Generates visual insights, including:
  - Sentiment distributions.
  - Trends over time for positive and negative tweets.
  - Top keywords for positive and negative sentiment.
  - Co-occurrence heatmaps for keywords.
  - Word clouds for sentiment categories.

---

## Example Output
1. **CSV Output**: 
   - The `sentiment_results.csv` file includes:
     - Cleaned tweet text.
     - Sentiment labels from FinBERT and Logistic Regression.
   - Example:
     ```
     text                | timestamp       | clean_text         | finbert_label | logistic_label
     "AAPL is soaring!"  | 2023-12-01 10:30| aapl is soaring    | 0             | 0
     ```

2. **Plots**: 
   - Sentiment distribution bar charts.
   - Sentiment trends over time.
   - Keyword bar plots and word clouds.

---

## Notes and Tips
1. **GPU Acceleration**:
   - If your system has a GPU, the FinBERT predictions will run significantly faster.
   - Ensure `torch` is set to utilize the GPU (`device = "cuda"`).

2. **Dataset Size**:
   - For testing purposes, a sample of 1,000 tweets is used. Adjust the `sampled_data` line to process a larger dataset if desired.

3. **Custom Dataset**:
   - If you have your own dataset, ensure the format matches the required columns (`text`, `timestamp`).

