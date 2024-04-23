# Movie Recommendation System using Collaborative Filtering

## Overview

This project implements a movie recommendation system based on Collaborative Filtering. Collaborative Filtering is a technique commonly used in recommendation systems where the system predicts a user's preferences or interests based on the preferences or interests of similar users.

## Project Structure

- `ratings.csv`: Contains ratings provided by users for different movies.
- `movies.csv`: Contains information about various movies, including their titles and genres.
- `Notebook.ipynb`: Jupyter Notebook containing the implementation of the recommendation system.

## Dependencies

- numpy
- pandas
- fuzzywuzzy
- matplotlib
- scikit-learn

## Usage

1. Ensure you have the required Python libraries installed.
2. Place the `ratings.csv` and `movies.csv` files in the same directory as your Python script.
3. Open the `Notebook.ipynb` file in Jupyter Notebook or any compatible environment.
4. Run the cells in the notebook to load and preprocess the data, train the model, and implement the recommendation engine.
5. Call the `movie_recommender_engine()` function with the desired movie name as input to get recommendations.

## Note

- The recommendation system uses Collaborative Filtering, which relies on user ratings. Hence, it might not work effectively for new or unpopular movies with few ratings.
<hr>