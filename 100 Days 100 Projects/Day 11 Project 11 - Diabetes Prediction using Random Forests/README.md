# Diabetes Prediction using Random Forests

This project demonstrates how to predict diabetes using Random Forests, a popular machine learning algorithm. The goal is to build a predictive model that can accurately classify individuals as diabetic or non-diabetic based on various features.

## Project Overview

The project is implemented in a Jupyter Notebook (Notebook.ipynb) and consists of the following steps:

1. **Importing Necessary Libraries**: The required libraries, including NumPy, pandas, Matplotlib, seaborn, and scikit-learn, are imported.

2. **Importing the Dataset**: The dataset containing information about individuals' health attributes is loaded from a CSV file ("diabetes_prediction_dataset.csv").

3. **Data Preprocessing**: The dataset is preprocessed by encoding categorical variables to numerical values using LabelEncoder, and converting categorical variables to dummy variables using one-hot encoding.

4. **Feature Selection**: The dataset is divided into features (X) and the target variable (y).

5. **Data Splitting**: The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

6. **Feature Scaling**: The features are scaled using StandardScaler to standardize the range of feature values.

7. **Model Training**: A Random Forest classifier is trained on the training set using RandomForestClassifier from scikit-learn.

8. **Model Evaluation**: The trained model is evaluated on the test set using confusion matrix and accuracy score metrics from scikit-learn.

9. **Visualizing the Confusion Matrix**: The confusion matrix is visualized using seaborn's heatmap.

10. **Saving Results**: The actual and predicted values are saved to a CSV file ("predictions.csv").

## Usage

1. Download the Notebook.ipynb file and the dataset ("diabetes_prediction_dataset.csv") to your local machine.

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
- seaborn
- scikit-learn

Install the required dependencies using pip.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<hr>