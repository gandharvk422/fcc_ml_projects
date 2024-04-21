# Breast Cancer Classification using Support Vector Machines

This project implements a Support Vector Machine (SVM) model for the classification of breast cancer using Python's `scikit-learn` library. The SVM is trained to classify breast cancer tumors as either malignant (M) or benign (B) based on features extracted from diagnostic images.

## Overview
The code consists of the following main steps:

1. **Importing the Libraries**: Necessary libraries such as numpy, pandas, scikit-learn's SVM, matplotlib, and others are imported.

2. **Load the Data**: The breast cancer dataset is loaded from a CSV file named "breast-cancer.csv".

3. **Preprocessing**: The data is preprocessed by removing unnecessary columns and standardizing the features using `StandardScaler`.

4. **Splitting the Data**: The preprocessed data is split into training and testing sets using `train_test_split`.

5. **Training the SVM Model**: An SVM model with a linear kernel is trained on the training set.

6. **Prediction**: The trained model is used to predict the diagnosis for a new data point and the test set.

7. **Evaluation**: Confusion matrix and accuracy score are calculated to evaluate the performance of the model.

8. **Visualization**: The confusion matrix is visualized using matplotlib.

9. **Saving Predictions**: The actual and predicted values from the test set are saved to a CSV file named "predictions.csv".

## Usage
To use this code, ensure you have Python installed along with the necessary libraries specified in the code. You can run the code in any Python environment or IDE.

```bash
jupyter notebook Notebook.ipynb
```

## Dataset
The dataset used in this project is "breast-cancer.csv". It contains diagnostic features for breast cancer tumors, including features extracted from digitized images of a breast mass. The target variable is "diagnosis", which indicates whether the tumor is malignant (M) or benign (B).

## Dependencies
- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
<hr>