# Project: Spam Email Detection using Naive Bayes Classifier

## Introduction
This project aims to detect spam emails using a Naive Bayes Classifier. The classifier is trained on a dataset containing labeled email messages, distinguishing between 'ham' (non-spam) and 'spam' emails.

## Code Overview

### 1. Data Loading and Preprocessing
- The dataset is loaded from a CSV file using pandas.
- The 'v1' column, representing labels ('ham' or 'spam'), is encoded as 0 for 'ham' and 1 for 'spam'.
- Text data from the 'v2' column is used as features.

### 2. Feature Extraction
- The text data is transformed into numerical features using the TF-IDF vectorizer.
- TF-IDF (Term Frequency-Inverse Document Frequency) is a technique to quantify the importance of words in documents.

### 3. Splitting Data
- The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.
- The training set contains 80% of the data, while the testing set contains 20%.

### 4. Model Training
- A Multinomial Naive Bayes classifier is initialized.
- The classifier is trained on the training data using the `fit` method.

### 5. Prediction and Evaluation
- The trained model is used to predict labels for the testing data.
- Confusion matrix, accuracy, and classification report are printed to evaluate the model's performance.

## Conclusion
The Naive Bayes Classifier trained on this dataset demonstrates the effectiveness of using machine learning techniques for spam email detection. The model achieves a high accuracy in distinguishing between spam and non-spam emails, as indicated by the evaluation metrics.
<hr>