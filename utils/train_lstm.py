"""
This script trains and saves a RNN w/ LSTM model for binary (positive vs. negative)
context-free text sentiment classification. Make sure to have the latest alpha 2.0
version of the TensorFlow to correctly load, train, and save the model.
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
print("Number of unique words in mapping dictionary", len(vocab_to_int.keys()))

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

#### TRAINING - VALIDATION DATASET SPLIT ####
# Shuffle in a corresponding manner
tmp = list(zip(EQUALIZED_ENCODED_REVIEWS, ALL_LABELS))
random.shuffle(tmp)
EQUALIZED_ENCODED_REVIEWS[:], ALL_LABELS[:] = zip(*tmp) # * for unpacking

# Specify split fraction: Train = 80% | Validation = 20%
split_constant = 0.8
split_index = int(len(ALL_REVIEWS)*split_constant)
X_train, Y_train = EQUALIZED_ENCODED_REVIEWS[:split_index], ALL_LABELS[:split_index]
X_validation, Y_validation = EQUALIZED_ENCODED_REVIEWS[split_index:], ALL_LABELS[split_index:]

#### CHECKING SHAPES ####
print("Total number of avaiable Data: ", len(ALL_REVIEWS))
print("Training Data Shape: ", np.shape(X_train))
print("Training Labels Shape: ", np.shape(Y_train))
print("Validation Data Shape: ", np.shape(X_validation))
print("Validation Labels Shape: ", np.shape(Y_validation))

#### TRAINING AND EVALUATION ####
# Create new model class
VOCABULARY_SIZE = len(vocab_to_int.keys())
model = RNN_LSTM(sequence_length=SEQUENCE_LENGTH,
                 vocabulary_size=VOCABULARY_SIZE)

# Specify the training configuration with the compile step
# Change loss function to 'categorical_crossentropy' for multi-label classification
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0003),
              metrics=['accuracy'])
# print(model.summary())

# Run below only if script is the primary script being executed.
if __name__ == "__main__":
    if args.config == "load":
        # Load previously trained Keras model
        # Load json and create model
        # json_file = open(args.load_folder + args.load_model_name + '.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = tf.keras.models.model_from_json(loaded_model_json)

        # This initializes the variables used by the optimizers, as well as any stateful metric variables
        model.train_on_batch(x=np.array(X_train[:1], dtype=np.int32),
                             y=np.array(Y_train[:1], dtype=np.int32))
        # Load weights into new model
        model.load_weights(args.load_folder + args.load_model_name)
        print("Loaded model from disk...")

    # TODO: Figure out a way to save & load subclass Keras models

    # Define callbacks for supervision of training process
    callbacks = [
      # Interrupt training if `val_loss` stops improving for over 1 epoch -> have to add validation data
      tf.keras.callbacks.EarlyStopping(patience=1, monitor='val_loss'),
      # tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
      # tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
      # Write TensorBoard logs to `./logs` directory
      tf.keras.callbacks.TensorBoard(log_dir='../logs')
    ]

    history = model.fit(x=np.array(X_train, dtype=np.int32), # NumPy arr w/ shape (num_samples, num_features)
                        y=np.array(Y_train, dtype=np.int32), # NumPy arr w/ shape (num_samples, )
                        batch_size=None,
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=(np.array(X_validation),
                                         np.array(Y_validation)))

    # Evaluate and print test loss and accuracy
    validation_loss, validation_accuracy = model.evaluate(x=np.array(X_validation),
                                                          y=np.array(Y_validation))

    print('Validation Loss: {}'.format(validation_loss))
    print('Validation Accuracy: {}'.format(validation_accuracy))

    # Save trained Keras Model for faster prediction
    # Serialize model to JSON
    # model_json = model.to_json()
    # with open(args.save_folder + args.save_model_name + '.json', "w") as json_file:
        # json_file.write(model_json)
    # Serialize weights to HDF5
    # model.save_weights(args.save_folder + args.save_model_name + '.h5')
    model.save_weights(args.save_folder + args.save_model_name, save_format='tf')
    print("Saved model to disk...")

    #### TESTING AND EXAMPLES ####

    # Predict on a sample text without padding
    # sample_pred_text = "I don't really trust DogeCoin. "
    # sample_pred_text += "The name is cute, but is doomed to be replaced by another cryptocurrency!"
    # predictions = sample_predict(model,
                                 #sample_pred_text,
                                 #vocab_to_int,
                                 #SEQUENCE_LENGTH)

    # print(predictions) # < 0.5 : Negative sentiment, >= 0.5 : Positive sentiment
