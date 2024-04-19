# House Price Prediction with Random Forest

## Overview

This project focuses on predicting house prices using a Random Forest regression model. It utilizes the Iowa housing dataset to train and evaluate the model's performance. The goal is to accurately predict the sale prices of houses based on various features such as lot area, year built, square footage, and number of rooms, among others.

## Dependencies

- pandas
- scikit-learn

## Getting Started

To run the project, make sure you have Python installed on your machine along with the required dependencies mentioned above.

1. Clone this repository to your local machine.
2. Ensure you have the `train.csv` and `test.csv` files in the project directory.
3. Open the project directory in your terminal or command prompt.
4. Run the `house_price_prediction.py` script using the command `python house_price_prediction.py`.
5. Once the script completes execution, you will find the predicted house prices stored in the `submission.csv` file.

## Project Structure

The project consists of the following files:

- `train.csv`: The training dataset containing features and sale prices of houses.
- `test.csv`: The test dataset used for making predictions.
- `house_price_prediction.py`: The Python script containing the code for data preprocessing, model training, and prediction generation.
- `submission.csv`: The output file containing the predicted sale prices for houses in the test dataset.

## Usage

- Modify the `features` list in the `house_price_prediction.py` script to include additional features for model training.
- Experiment with different machine learning algorithms and hyperparameters to improve prediction accuracy.
- Explore feature engineering techniques to enhance the model's performance.
- Visualize the data and model predictions to gain insights into the relationships between features and sale prices.

## Acknowledgments

This project is based on the Kaggle competition "House Prices: Advanced Regression Techniques." Special thanks to the Iowa Housing dataset contributors and the scikit-learn development team for providing the tools necessary for machine learning tasks.
<hr>