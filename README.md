# IMDb Movie Reviews Sentiment Analysis
This project focuses on sentiment analysis of IMDb movie reviews using various deep learning models. The steps involved include data preprocessing, building an embedding matrix using GloVe, and training models using Keras. Below is a detailed breakdown of each step.

## Table of Contents
1. Dataset
2. Data Preprocessing
3. Word Embedding
4. Model Training\
    Simple Neural Network\
    Convolutional Neural Network (CNN)\
    Long Short-Term Memory (LSTM)\
5. Evaluation
6. Predictions
7. Requirements
8. Usage
## Dataset
The dataset used for this project is the IMDb Movie Reviews dataset, which contains 50,000 movie reviews labeled as either positive or negative.

Source: [IMDb Movie Reviews dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)\
## Data Preprocessing
Loading the dataset: The dataset is loaded and split into training and testing sets.\
Cleaning the reviews: Special characters, numbers, and other non-alphabetic characters are removed from the reviews.\
Label encoding: Sentiment labels are converted to numerical values: positive (1) and negative (0).\
## Word Embedding
Import GloVe Word Embedding: The GloVe (Global Vectors for Word Representation) embeddings are used to build the embedding dictionary.\
Embedding Matrix: An embedding matrix is constructed for the words in our corpus using the GloVe vectors.\
## Model Training
Three different models are trained using Keras to perform sentiment analysis:\

### Simple Neural Network
A basic neural network model with an embedding layer followed by dense layers.

### Convolutional Neural Network (CNN)
A CNN model is used to capture local features in the text data, utilizing convolutional layers.

### Long Short-Term Memory (LSTM)
An LSTM model is trained to capture long-term dependencies in the text data, making use of LSTM layers.

## Evaluation
The performance of each model is evaluated using accuracy, precision, recall, and F1-score on the test dataset.

## Predictions
The best performing model is used to perform predictions on new, real IMDb movie reviews.

## Requirements
* Python
* Keras
* Pandas
* Numpy
* Scikit-learn
* TensorFlow
* nltk
