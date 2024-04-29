# Customer Churn Prediction using XGBoost

This project demonstrates how to predict customer churn using XGBoost, a powerful gradient boosting algorithm. The goal is to build a predictive model that can accurately classify customers as churners or non-churners based on various features from a Telco company dataset.

## Project Overview

The project is implemented in a Jupyter Notebook (Notebook.ipynb) and consists of the following steps:

1. **Importing Necessary Libraries**: The required libraries, including NumPy, pandas, Matplotlib, scikit-learn, and XGBoost, are imported.

2. **Loading the Dataset**: The Telco customer churn dataset ("Telco-Customer-Churn.csv") is loaded into a pandas DataFrame.

3. **Data Preprocessing**: The dataset is preprocessed by converting the `TotalCharges` column to numeric and handling any errors that might occur during the conversion. Missing values in the `TotalCharges` column are filled with zeros. 

4. **Feature Selection**: The features are defined as all columns except the target variable (`Churn`).

5. **Target Variable Encoding**: The target variable (`Churn`) is encoded into numerical values using LabelEncoder.

6. **Feature Encoding**: Categorical variables in the feature set are converted into dummy/indicator variables using one-hot encoding.

7. **Data Splitting**: The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

8. **Model Training**: An XGBoost classifier is trained on the training set using XGBClassifier from the XGBoost library.

9. **Model Evaluation**: The trained model is evaluated on the test set using confusion matrix and accuracy score metrics from scikit-learn. The confusion matrix is visualized using ConfusionMatrixDisplay from scikit-learn.

10. **Saving Results**: The actual and predicted values are saved to a CSV file ("predictions.csv").

## Usage

1. Download the Notebook.ipynb file and the dataset ("Telco-Customer-Churn.csv") to your local machine.

2. Install Jupyter Notebook if you haven't already:

3. Open a terminal or command prompt, navigate to the directory containing the Notebook.ipynb file, and run the following command:
    `pip install notebook`

4. This will open the Jupyter Notebook interface in your web browser. Click on the Notebook.ipynb file to open it.

5. Execute each cell in the notebook sequentially to run the code and see the results.

## Requirements

- Jupyter Notebook
- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- XGBoost

Install the required dependencies using pip:

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<hr>