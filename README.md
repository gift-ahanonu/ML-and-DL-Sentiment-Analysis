# Sentiment Analysis of Tweets – ML vs Deep Learning

This project presents a **comparative study of traditional machine learning and deep learning approaches for sentiment classification of tweets**. The aim was to evaluate the strengths, limitations, and real-world applicability of models like Naïve Bayes, Logistic Regression, Random Forest, SVM, and an LSTM with Attention mechanism and GloVe embeddings.

>  Developed by Gift Ahanonu  

##  Project Overview

Social media platforms generate vast amounts of text data daily. Understanding the sentiment in these posts can support decision-making in business, government, and social research. This project explores the ability of various ML and DL models to classify tweets into **positive**, **neutral**, and **negative** categories.


## Contents

- `LSTM.ipynb` – Deep learning implementation using BiLSTM + Attention + GloVe
- `Naive+Logistic+Random.ipynb` – Traditional ML models: Naïve Bayes, Logistic Regression, Random Forest
- `SVM.ipynb` – SVM implementation
- `Tweets.csv` – Dataset from Kaggle

## Key Highlights

-  **Dataset**: 27,481 tweets (Kaggle), with preprocessing including cleaning, lemmatisation, and tokenisation.
-  **ML Models**: TF-IDF + Naïve Bayes, Logistic Regression, Random Forest, and SVM.
-  **DL Model**: BiLSTM + Attention with GloVe embeddings (200-dimensional).
-  **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices.

---

## 📈 Results Summary

| Model               | Accuracy | Notes |
|--------------------|----------|-------|
| Naïve Bayes        | 0.640    | Struggled with neutral sentiment |
| Logistic Regression| 0.687    | Balanced but struggled with context |
| Random Forest      | 0.680    | Strong on positives, risk of overfitting |
| SVM                | 0.699    | Best traditional model |
| LSTM + Attention   | 0.713    | Highest accuracy but poor ROC-AUC (~0.5) |

> Neutral sentiment detection remained a major challenge across all models.

---

## Future Improvements

- Fine-tune LSTM hyperparameters to reduce overfitting.
- Experiment with Transformer-based models like **BERT** or **RoBERTa**.
- Enhance class balancing and contextual embedding for improved neutral classification.

---

## Credits - Dataset

M Yasser H (2022). Twitter Tweets Sentiment Dataset. [online] Kaggle.com.
Available at: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentimentdataset/data. 
