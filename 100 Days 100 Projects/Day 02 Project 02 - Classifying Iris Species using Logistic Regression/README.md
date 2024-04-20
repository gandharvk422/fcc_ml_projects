# Classifying Iris Species using Logistic Regression

This project aims to classify Iris flower species based on their sepal and petal dimensions using logistic regression, a popular classification algorithm.

## Overview

The Iris dataset is a classic dataset commonly used for practicing classification algorithms. It contains 150 samples of Iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The goal of this project is to build a logistic regression model that can accurately predict the species of Iris flowers based on these features.

## Dataset

The Iris dataset used in this project is a part of the scikit-learn library. It consists of 150 samples, where each sample contains four features and a target label indicating the species of the Iris flower (setosa, versicolor, or virginica).

## Methodology

1. **Data Preprocessing**: The dataset is split into training and testing sets. Feature scaling is applied to standardize the feature values.

2. **Model Training**: Logistic regression model is trained using the training data. Logistic regression is a binary classification algorithm that predicts the probability of a sample belonging to a particular class.

3. **Model Evaluation**: The trained model is evaluated using the testing data. Performance metrics such as confusion matrix and accuracy score are calculated to assess the model's classification accuracy.

4. **Predictions**: The trained model is then used to make predictions on new, unseen data. The actual and predicted values are saved to a CSV file for further analysis.

## Dependencies

This project requires the following dependencies:
- Python (>=3.6)
- scikit-learn
- pandas

You can install the dependencies using pip:


## Usage

To run the project:
1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the Python Notebook `Notebook.ipynb`.
4. Check the generated CSV file `iris_predictions.csv` for actual and predicted values.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<hr>