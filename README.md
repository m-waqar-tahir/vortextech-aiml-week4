# Sentiment Analysis on IMDB Movie Reviews

Vortex Tech AI & ML Internship 2026 — Week 4 (Capstone)

## Overview

A binary sentiment classification model built on 50,000 IMDB movie reviews,
combining text preprocessing, TF-IDF feature extraction, and two classifier
comparisons — the capstone project for the internship track.

## Dataset

`IMDB-Dataset.csv` — 50,000 movie reviews, two columns: `review` (raw text)
and `sentiment` (positive/negative). Perfectly balanced: 25,000 positive and
25,000 negative reviews, no missing values. Source: IMDB Large Movie Review
Dataset (Maas et al., ACL 2011), a standard sentiment analysis benchmark.

## Files

- `sentiment_analysis_imdb.ipynb` — main notebook, every code cell preceded
  by a titled markdown description
- `IMDB-Dataset.csv` — dataset used by the notebook
- `Week4_Sentiment_Analysis_Full_Guide.txt` — step-by-step explanation of
  every code cell, including the reasoning behind the text cleaning order
  and model comparison
- `requirements.txt` — Python libraries required to run the notebook

## What the notebook covers

- Inspecting class balance (confirmed perfectly balanced, unlike Week 2)
- Text cleaning: lowercasing, removing leftover HTML tags from the original
  scrape, stripping punctuation and numbers, collapsing whitespace
- Converting cleaned text into 5,000 TF-IDF features with English stop
  words removed
- An 80/20 stratified train-test split
- Training and comparing two models: Logistic Regression and Multinomial
  Naive Bayes
- Evaluating both with accuracy and F1-score
- A confusion matrix visualization
- Testing the trained model on 3 original example sentences
- A final written summary covering approach, results, and an honest
  limitation

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```
   jupyter notebook sentiment_analysis_imdb.ipynb
   ```
3. Run all cells in order.

## Key findings

- Logistic Regression: accuracy 0.8885, F1-score 0.8895
- Multinomial Naive Bayes: accuracy 0.8531, F1-score 0.8540 — Logistic
  Regression was chosen as the stronger model based on this comparison
- Because the dataset is perfectly balanced, accuracy and F1-score landing
  close together is expected and is itself a sign of healthy evaluation —
  a direct contrast with Week 2, where a large gap between accuracy and
  minority-class metrics revealed a class imbalance problem
- Custom sentence test: "This was the best experience ever" → Positive
  (98.1% confidence); "This was a complete waste of time" → Negative
  (99.7% confidence); "The movie was okay, not great but not terrible
  either" → Negative (89.0% confidence)

## Limitation

The model struggles with mixed or neutral sentiment. The third custom test
sentence above is genuinely ambivalent, but was confidently classified as
Negative. Since the model was trained only on clearly positive or clearly
negative reviews with no neutral category, it has no mechanism for
expressing uncertainty, and appears to weight strong individual words like
"terrible" heavily even when surrounding context softens their meaning.
Sarcasm is a closely related limitation not directly tested here: a
sarcastic sentence with positive-sounding words but negative intended
meaning would likely be misread the same way, since a bag-of-words TF-IDF
approach has no way to detect tone or word order. A model that captures
context, such as an LSTM or transformer-based model, would likely handle
both cases better.

## Status

✅ Completed

## Author

Muhammad Waqar Tahir
Vortex Tech AI & ML Internship 2026 — Week 4
