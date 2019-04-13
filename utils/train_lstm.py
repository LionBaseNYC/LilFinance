"""
This script trains and saves a RNN w/ LSTM model for binary (positive vs. negative)
text sentiment classification.
Inspired from https://github.com/adeshpande3/LSTM-Sentiment-Analysis
Data utilized for training the sentiment analysis model is taken from O'Reilly
tutorial on sentiment analysis with LSTMs in Tensorflow that can be found here:
https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow

Usage:
  # Load previously saved model and save new version (if preferred, overwrite):
  python train_lstm.py --config=load --load_folder=../saved_models/ --load_model_name=model
                       --save_folder=../saved_models/ --save_model_name=model
  # Train new model and save (if prefferd, overwrite):
  python train_lstm.py --save_folder=../saved_models/ --save_model_name=model

If you didn't touch the initial folder structure of the project, running python
--config=load will simply load the previous model file, train, and overwrite.
"""

# Standard imports
import os
import numpy as np
import tensorflow as tf
# Imports for data visualization
import pandas as pd
import matplotlib.pyplot as plt
# Imports for shuffling
import random
# Imports for sentiment analyis utilities
from sentiment_analysis_utils import *
# Import recurrent neural network model
from sentiment_lstm_model import RNN_LSTM
# Import dictionary checker library
import enchant # pip install pyenchant
ENGLISH_DICTIONARY = enchant.Dict("en_US")

# Argument specification
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=None,
                    help="Specify 'load' to load existing model file, else leave empty.")
parser.add_argument("--load_folder",
                    default="../saved_models/",
                    help="Name of the directory of the to-be loaded model file")
parser.add_argument("--load_model_name",
                    default="model",
                    help="Name of the to-be loaded model file")
parser.add_argument("--save_folder",
                    default="../saved_models/",
                    help="Name of the directory of the to-be saved model file")
parser.add_argument("--save_model_name",
                    default="model",
                    help="Name of the to-be saved model file")
args = parser.parse_args()

# Data path and configurations
POSITIVE_REVIEWS_PATH = "../sentiment_data/positiveReviews"
POSITIVE_REVIEWS = []
POSITIVE_LABEL = 1

NEGATIVE_REVIEWS_PATH = "../sentiment_data/negativeReviews"
NEGATIVE_REVIEWS = []
NEGATIVE_LABEL = 0

#### DATA READING ####
# Read in positive data
for filename in os.listdir(POSITIVE_REVIEWS_PATH):
    arr = []
    if filename.endswith(".txt"):
        review = open(POSITIVE_REVIEWS_PATH +  "/" + filename, "r")
        for line in review:
            POSITIVE_REVIEWS.append(line)

# Read in negative data
for filename in os.listdir(NEGATIVE_REVIEWS_PATH):
    arr = []
    if filename.endswith(".txt"):
        review = open(NEGATIVE_REVIEWS_PATH +  "/" + filename, "r")
        for line in review:
            NEGATIVE_REVIEWS.append(line)

#### DATA PROCESSING ####
# Concatanate all reviews under single list
ALL_REVIEWS = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
LAST_POSITIVE_INDEX = len(POSITIVE_REVIEWS) - 1

# Remove punctuations and convert to lowercase
CLEANED_ALL_REVIEWS = remove_punctuation_and_lower(ALL_REVIEWS)

#### TOKENIZATION - MAPPING ####
vocab_to_int = get_tokenization_map(CLEANED_ALL_REVIEWS)
# print(vocab_to_int) # debugging

#### TOKENIZATION - ENCODING ####
ENCODED_REVIEWS = get_encoding(CLEANED_ALL_REVIEWS, vocab_to_int)

#### OPTIONAL: UNCOMMENT TO VISUALIZE AND ANALYZE THE WORD LENGTH OF REVIEWS ####
"""
positive_text_length = [len(text_map) for text_map in ENCODED_REVIEWS[:LAST_POSITIVE_INDEX+1]]
positive_series = pd.Series(positive_reviews_length)
print(positive_series.describe()) # print summary statistics
positive_series.hist()
plt.show()

negative_reviews_length = [len(w_map) for w_map in ENCODED_REVIEWS[LAST_POSITIVE_INDEX+1:]]
negative_series = pd.Series(negative_reviews_length)
print(negative_series.describe()) # print summary statistics
negative_series.hist()
plt.show()
"""

#### PADDING/TRUNCATING THE REMAINING DATA ####
SEQUENCE_LENGTH = 230
EQUALIZED_ENCODED_REVIEWS = pad_or_truncate(ENCODED_REVIEWS, SEQUENCE_LENGTH)

#### CREATE LABELS FOR THE DATA ####
POSITIVE_LABELS = np.array([POSITIVE_LABEL for review in ENCODED_REVIEWS[:LAST_POSITIVE_INDEX+1]])
NEGATIVE_LABELS = np.array([NEGATIVE_LABEL for review in ENCODED_REVIEWS[LAST_POSITIVE_INDEX+1:]])
ALL_LABELS = np.concatenate((POSITIVE_LABELS, NEGATIVE_LABELS))
#### TRAINING - TEST DATASET SPLIT ####
# Shuffle in a corresponding manner
tmp = list(zip(EQUALIZED_ENCODED_REVIEWS, ALL_LABELS))
random.shuffle(tmp)
EQUALIZED_ENCODED_REVIEWS[:], ALL_LABELS[:] = zip(*tmp) # * for unpacking

# Specify split fraction: Train = 80% | Test = 20%
split = 0.8
X_train, Y_train = EQUALIZED_ENCODED_REVIEWS[:int(len(ALL_REVIEWS)*split)], ALL_LABELS[:int(len(ALL_LABELS)*split)]
X_test, Y_test = EQUALIZED_ENCODED_REVIEWS[int(len(ALL_REVIEWS)*split):], ALL_LABELS[int(len(ALL_LABELS)*split):]

#### CHECKING SHAPES ####
print("Total number of avaiable Data: ", len(ALL_REVIEWS))
print("Training Data Shape: ", np.shape(X_train))
print("Training Labels Shape: ", np.shape(Y_train))
print("Testing Data Shape: ", np.shape(X_test))
print("Testing Labels Shape: ", np.shape(Y_test))

####################################################
model = None

if args.config == "load":
    # Load previously trained Keras model
    # Load json and create model
    json_file = open(args.load_folder + args.load_model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(args.load_folder + args.load_model_name + '.h5')
    print("Loaded model from disk...")

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

else:
    # Create new model class
    VOCAB_SIZE = len(vocab_to_int.keys())
    model = RNN_LSTM(VOCAB_SIZE).get_model()


history = model.fit(x=np.array(X_train), # NumPy arr w/ shape (num_samples, num_features)
                    y=np.array(Y_train), # NumPy arr w/ shape (num_samples, )
                    batch_size=None,
                    epochs=1)

# Evaluate and print test loss and accuracy
test_loss, test_acc = model.evaluate(x=np.array(X_test),
                                     y=np.array(Y_test))

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# Save trained Keras Model for faster prediction
# Serialize model to JSON
model_json = model.to_json()
with open(args.save_folder + args.save_model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights(args.save_folder + args.save_model_name + '.h5')
print("Saved model to disk...")

# Predict on a sample text without padding
sample_pred_text = "I don't really trust DogeCoin. "
sample_pred_text += "The name is cute, but is doomed to be replaced by another cryptocurrency!"
predictions = sample_predict(model,
                             sample_pred_text,
                             vocab_to_int,
                             SEQUENCE_LENGTH)
print(predictions)
