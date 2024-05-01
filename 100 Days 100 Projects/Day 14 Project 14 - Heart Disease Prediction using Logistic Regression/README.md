# Heart Disease Detection using Logistic Regression

This project aims to predict the likelihood of heart disease in individuals within the next 10 years using logistic regression. The logistic regression model is trained on the Framingham Heart Study dataset, which contains various health-related features such as age, gender, cholesterol levels, blood pressure, and smoking habits.

## Notebook.ipynb

The notebook `Notebook.ipynb` contains the Python code for the project. Here's a breakdown of the main sections:

### 1. Importing Libraries
- Importing necessary libraries such as NumPy, Pandas, scikit-learn modules, and Matplotlib.

### 2. Loading the Data
- Loading the Framingham Heart Study dataset (`framingham.csv`) into a Pandas DataFrame.

### 3. Data Preprocessing
- Handling missing values using SimpleImputer.
- Splitting the dataset into training and test sets.
- Performing feature scaling using StandardScaler.

### 4. Model Training
- Training a logistic regression model on the training data.

### 5. Single Prediction
- Predicting the likelihood of heart disease for a single individual.

### 6. Model Evaluation
- Predicting the test set results.
- Evaluating the model's performance using confusion matrix and accuracy score.
- Visualizing the confusion matrix.

### 7. Saving Predictions
- Creating a DataFrame of actual and predicted values and saving it to a CSV file (`predictions.csv`).

## Usage
To run the notebook and perform heart disease detection:
1. Ensure you have Python installed along with the required libraries listed in the notebook.
2. Download the `framingham.csv` dataset.
3. Open and run the `Notebook.ipynb` in a Jupyter Notebook or any compatible environment.

## Dataset
The Framingham Heart Study dataset (`framingham.csv`) used in this project contains anonymized patient data collected from the Framingham Heart Study.

## Author
This project is authored by [Your Name].

## License
This project is licensed under the [MIT License](LICENSE).
<hr>
