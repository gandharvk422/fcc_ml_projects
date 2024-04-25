# Image Classification using Convolutional Neural Networks (CNNs)

This project demonstrates image classification using Convolutional Neural Networks (CNNs) implemented in TensorFlow. The dataset used for training and testing the model is CIFAR-10, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Overview

The project includes a Jupyter Notebook `Notebook.ipynb` that walks through the entire process from loading the CIFAR-10 dataset to training a CNN model for image classification. Here's a brief overview of the steps covered in the notebook:

1. Importing necessary libraries including TensorFlow, scikit-learn, and Matplotlib.
2. Loading and preprocessing the CIFAR-10 dataset.
3. Splitting the dataset into training, validation, and test sets.
4. Defining a CNN model architecture using TensorFlow's Keras API.
5. Compiling the model with appropriate loss function, optimizer, and metrics.
6. Training the model on the training set and validating it on the validation set.
7. Evaluating the trained model on the test set to measure its performance.
8. Visualizing the training and validation accuracy over epochs.
9. Performing inference on sample images to demonstrate the model's predictions.

## Files

- `Notebook.ipynb`: Jupyter Notebook containing the code for the project.
- `airplane.jpeg`: Sample image of an airplane used for inference.
- `dog.jpg`: Sample image of a dog used for inference.

## Instructions

To run the code in `Notebook.ipynb`, make sure you have the following prerequisites installed:

- Python 3.x
- TensorFlow
- scikit-learn
- Matplotlib

You can install these dependencies using pip:

```bash
pip install tensorflow scikit-learn matplotlib
```

## References

* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

* [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

* [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)


This README.md file provides an overview of the project, lists the files included, gives instructions for running the code, and provides references for further reading.
<hr>