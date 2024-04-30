# Movie Genre Classification using Text Classification

This project aims to classify movie genres based on their descriptions using text classification techniques. The dataset consists of movie descriptions and their corresponding genres.

## Notebook Overview

The `Notebook.ipynb` file contains the code for the entire project, including data loading, exploratory data analysis (EDA), data preprocessing, text cleaning, text vectorization using TF-IDF, model training using Multinomial Naive Bayes, evaluation of model performance, and making predictions on test data.

## Steps Covered in the Notebook:

1. **Data Loading and EDA**:
   - The training dataset (`train_data.csv`) and test dataset (`test_data.csv`) are loaded using pandas.
   - EDA is performed to understand the distribution of movie genres in the training data.

2. **Data Preprocessing and Text Cleaning**:
   - Text data is preprocessed and cleaned by removing special characters, URLs, stopwords, and non-alphabetic characters.
   - The cleaned text is tokenized and transformed into lowercase.

3. **Text Vectorization using TF-IDF**:
   - The TF-IDF vectorizer is initialized and applied to convert text data into numerical vectors.

4. **Model Training (Naive Bayes)**:
   - The dataset is split into training and validation sets.
   - A Multinomial Naive Bayes classifier is trained using the TF-IDF transformed data.
   
5. **Model Evaluation**:
   - The performance of the trained model is evaluated on the validation set using accuracy and classification report.

6. **Making Predictions**:
   - The trained model is used to make predictions on the test data.
   - The predicted genres are saved to a CSV file (`predicted_genres.csv`).

7. **Saving Predictions**:
   - A DataFrame containing actual and predicted genres is saved to a CSV file (`predictions.csv`).

## Requirements

- Python 3.x
- Libraries: pandas, seaborn, matplotlib, nltk, scikit-learn

## How to Use

1. Ensure you have all the required libraries installed.
2. Download the dataset files (`train_data.csv` and `test_data.csv`) and place them in a folder named `dataset`.
3. Run the `Notebook.ipynb` file step by step to execute the code and see the results.

Feel free to explore and modify the code according to your requirements!
<hr>