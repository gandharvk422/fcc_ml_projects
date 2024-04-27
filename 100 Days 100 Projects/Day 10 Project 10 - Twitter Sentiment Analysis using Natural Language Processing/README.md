# Twitter Sentiment Analysis using Natural Language Processing

This project demonstrates how to perform sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques. The goal is to classify tweets into positive and negative sentiments.

## Project Overview

The project is implemented in a Jupyter Notebook (Notebook.ipynb) and consists of the following steps:

1. **Importing Necessary Libraries**: The required libraries, including NLTK, NumPy, pandas, Matplotlib, and scikit-learn, are imported.

2. **Downloading Twitter Samples**: The Twitter samples are downloaded using the `twitter_samples` corpus from NLTK.

3. **Text Preprocessing**: The tweets are preprocessed by removing special characters, converting text to lowercase, tokenizing, stemming, and removing stopwords.

4. **Vectorization**: The preprocessed text data is vectorized using the CountVectorizer from scikit-learn, which converts text data into numerical features.

5. **Data Splitting**: The vectorized data is split into training and testing sets using scikit-learn's `train_test_split` function.

6. **Model Training**: A Gaussian Naive Bayes classifier is trained on the training set using scikit-learn's `GaussianNB`.

7. **Model Evaluation**: The trained model is evaluated on the test set using confusion matrix and accuracy score metrics from scikit-learn.

8. **Predictions**: The model makes predictions on the test set, and the actual and predicted values are printed and saved to a CSV file.

## Usage

1. Download the Notebook.ipynb file to your local machine.

2. Install Jupyter Notebook if you haven't already:

3. Open a terminal or command prompt, navigate to the directory containing the Notebook.ipynb file, and run the following command:

4. This will open the Jupyter Notebook interface in your web browser. Click on the Notebook.ipynb file to open it.

5. Execute each cell in the notebook sequentially to run the code and see the results.

## Requirements

- Jupyter Notebook
- Python 3
- NLTK
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install the required dependencies using pip:

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<hr>