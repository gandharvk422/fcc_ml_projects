# Credit Card Fraud Detection using Decision Trees

## Steps:

1. **Importing Necessary Libraries:** The required libraries such as NumPy, pandas, and scikit-learn modules for data processing, model training, and evaluation are imported.

2. **Loading the Data:** The credit card transaction data is loaded from the "creditcard.csv" file. 

3. **Splitting the Dataset:** The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

4. **Feature Scaling:** Standard scaling is applied to normalize the features.

5. **Training the Model:** A Decision Tree Classifier is instantiated with entropy as the criterion and is trained on the training data.

6. **Making Predictions:** The trained model is used to make predictions on the test set.

7. **Evaluation:** The confusion matrix and accuracy score are calculated to evaluate the model's performance.

8. **Creating a DataFrame for Actual and Predicted Frauds:** A DataFrame is created to compare actual and predicted fraud labels.

9. **Saving Predictions:** The predictions are saved to a CSV file named "predictions.csv".

## Files:

- **Notebook.ipynb:** Contains the Jupyter notebook with the code implementation.
- **creditcard.csv:** Dataset containing credit card transaction data.
- **predictions.csv:** CSV file where the predicted fraud labels are saved.

## Dependencies:

- Python 3.x
- NumPy
- pandas
- scikit-learn

## Usage:

1. Clone the repository.
2. Install the necessary dependencies.
3. Run the Notebook.ipynb file in a Jupyter environment.
<hr>