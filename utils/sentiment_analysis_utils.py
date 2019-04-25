#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 4 02:19:56 2019
Utilities included for use by Naive Bayes and RNN w/ LSTM model training and
data preparation, and also performing static sentiment analysis which is
roughly adapted from LionBase's Sentiment Analysis workshop.
"""
# Standard imports
import numpy as np
# Imports for getting online sentiment data
import requests
# Imports for punctuating removing
from string import punctuation
# Imports for tokenization
from collections import Counter
# Import dictionary checker library
import enchant # pip install pyenchant
ENGLISH_DICTIONARY = enchant.Dict("en_US")

# UTILITIES USED BY RNN w/ LSTM MODEL AND ITS TRAINING #
def remove_punctuation_and_lower(str_arr):
    """
    Function to convert the strings inside an array into lowercase format
    and remove punctuation.
    @param str_arr: array of texts, be it reviews or sentences
    @return clean_str_arr: cleaned string array
    """
    clean_str_arr = [None] * len(str_arr)
    for i in range(len(str_arr)):
        # Convert  text into lowercase format
        clean_str_arr[i] = str_arr[i].lower()
        # Remove punctuation from text
        clean_str_arr[i] = ''.join([c for c in clean_str_arr[i] if c not in punctuation])
    return clean_str_arr

def get_tokenization_map(str_arr, K = 1, strict = True):
    """
    Function to create a vocabulary-to-integer index mapping dictionary,
    where frequently occurring words are assigned to lower indexes.
    @param str_arr: array of texts, be it reviews or sentences
    @param K: most common words factor, decides how many words
           the mapping dictionary contains. Set to 1 for tokenization.
    @param strict: boolean value to decide whether we only want recognized
           English words in mapping dictionary (True), or everything (False)
    @return vocabulary_to_integer: mapping dictionary
    """
    all_words = []
    for i in range(len(str_arr)):
        # Break down into list of words
        for word in str_arr[i].split():
            if strict:
                if ENGLISH_DICTIONARY.check(word): # NOTE: This increases the runtime vastly
                    all_words.append(word)
            else:
                all_words.append(word)

    # Count all the words using Counter Method
    words_with_counts = Counter(all_words)
    count_sorted_words = words_with_counts.most_common(int(len(all_words)*K)) # get all words
    # print(str(count_sorted_words)) # debugging
    # Create mapping dictionary
    vocabulary_to_integer = {word:index+1 for index, (word,count) in enumerate(count_sorted_words)}
    # print(vocabulary_to_integer) # debugging
    return vocabulary_to_integer

def get_encoding(str_arr, vocabulary_to_integer, predicting=False):
    """
    Function to encode each word in each string in the array into its corresponding
    integer index defined in the passed tokenization mapping dictionary.
    @param str_arr: array of texts, be it reviews or sentences
    @param vocabulary_to_integer: tokenization mapping dictionary
    @param predicting: boolean value to alert us to check if text (array) is newly seen
    @return encoded_arr: array of encoded integer arrays
    """
    encoded_arr = []
    for text in str_arr:
        # If text is newly seen, check if mapping includes the passed words
        mapped = []
        corresponding = lambda x: vocabulary_to_integer[x] if x in vocabulary_to_integer.keys() else 0
        if predicting:
            mapped = [corresponding(word) for word in text.split()]
        else:
            mapped = [corresponding(word) for word in text.split()]
        encoded_arr.append(mapped)
    return encoded_arr

def pad_or_truncate(encoded_arr, sequence_length):
    """
    Function to pad the encoded array with 0s or truncate it,
    depending on its length with comparison to specified sequence length.
    @param encoded_arr: Encoded (using vocab-to-int map) text array
    @param sequence_length (int): normalizing length for reviews,
           also known as the number of steps for the LSTM layer,
           and also the number of features for each input in the model.
    @return features: padded OR truncated encoded array, will serve
            as a input for the model.
    """
    features = np.zeros((len(encoded_arr), sequence_length), dtype=int)

    for i, encoded_text in enumerate(encoded_arr):
        text_length = len(encoded_text)
        if text_length <= sequence_length:
            normalized_text = encoded_text + list(np.zeros(sequence_length - text_length, dtype=int))
        elif text_length > sequence_length:
            normalized_text = encoded_text[0:sequence_length]

        features[i,:] = np.array([int(id) for id in normalized_text])

    return list(features)

def sample_predict(model, X, vocabulary_to_integer, sequence_length):
    """
    Function to send a sample input and retrieve predictions from a
    Keras model only.
    @param model: Compiled Keras model class
    @param X: Input
    @param vocabulary_to_integer: Mapping
    @sequence_length: Sequence length
    @return predictions
    """
    clean_X = remove_punctuation_and_lower([X])
    encoded = get_encoding(clean_X, vocabulary_to_integer, predicting=True)
    equalized = pad_or_truncate(encoded, sequence_length)
    return model.predict(x=np.array(equalized))
    # NOTE: If equalized was a singular string, we could have
    # surrounded it with [ ] (square brackets) as np.array
    # asks for a list to convert to NumPy format

# UTILITIES USED BY STATIC SENTIMENT ANALYSIS METHODS #
def listify(text):
    """
    Function to convert a string text into a list of words.
    @param text: space seperated words
    @return words: list of lowercase words
    """
    words = text.lower().split(' ')

    punctuation = ['.', ',', ';', '?', '!']
    for i in range(len(words)):
        for mark in punctuation:
            if mark in words[i]:
                # Remove all punctuation marks for valid assessment
                words[i] = words[i].replace(mark, '')

    # Suffixes that can be appended to a word
    suffixes = ['eer', 'er', 'ion', 'ity', 'ment', 'ness', 'or', 'sion', 'ship', 'th',
                'able', 'ible', 'al', 'ant', 'ary', 'ful', 'ic', 'ious', 'ous', 'ive'
                'less', 'y', 'ed', 'en', 'er', 'ing', 'ize', 'ise', 'ly', 'ward', 'wise']
    # TODO: Utilize suffixes

    return words

def get_positive_negative_words():
    """
    Function to get positive and negative words based on Hu and Liu's
    sentiment analysis. Please refer to their following paper:
    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." Proceedings of the ACM
    SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-2004), Aug 22-25,
    2004, Seattle, Washington, USA.

    @return (positive_words, negative_words): tuple of two lists, representing +/- words
    """
    lists = []
    # URL's for positive and negative coded words lists
    for url in ('http://ptrckprry.com/course/ssd/data/positive-words.txt',
                'http://ptrckprry.com/course/ssd/data/negative-words.txt'):
        try:
            words = requests.get(url).content.decode('latin-1')
        except:
            print("There may be a connection issue, or the URL is no longer active!")
            break

        start_index = words.rfind(';') # get last occurence of ';', indicates start of list
        word_list = words[start_index:].split('\n') # split by new lines
        word_list = [word for word in word_list if len(word) > 1] # only include words with len > 1
        lists.append(word_list)

    # Return positive_words, negative_words
    return (lists[0], lists[1])

def compute_cross_sentiment(text, positive_words, negative_words):
    """
    Function to compute cross sentiment score based on the number of positive
    and negative words included.
    @arg text: either a list of words or a string with words seperated by spaces
    @arg positive_words: list containing words that are categorized as positive
    @arg negative_words: list containing words that are categorized as negative
    @return score: integer score, given by # positive occurences - # negative occurences
    """
    # If text is not a list, convert it to a list
    if not isinstance(text, list):
        text = listify(text)
        # TODO: remove punctuation marks

    # Keep counters for positive and negative word occurrences
    positive_occurrences = 0
    negative_occurrences = 0

    for word in text:
        if word in positive_words:
            positive_occurrences += 1
        elif word in negative_words:
            negative_occurrences += 1

    score = positive_occurrences - negative_occurrences
    return score

def get_emotion_dictionary():
    """
    Function to get a emotion dictionary, check emotions_file for more detail.
    @return emotion_dict: dictionary that accepts lowercase words as keys, and
            contains a list of emotions correlated to the word as values
    """
    emotions_file = "../sentiment_data/NRC-emotion-lexicon-wordlevel-alphabetized.txt"
    emotion_dict = {}
    with open(emotions_file,'r') as file:
        for line in file:
            line = line.strip().split('\t') # split by tabs
            if int(line[2]) == 1: # if 1,
                if emotion_dict.get(line[0]): # if word already exists in dictionary,
                    emotion_dict[line[0]].append(line[1]) # append new emotion
                else: # if word doesn't exist in dictionary,
                    emotion_dict[line[0]] = [line[1]] # create new list with emotion
    return emotion_dict


def compute_emotion_sentiment(text, emotion_dict):
    """
    Function that sets up, and returns a normalized result dictionary by
    analyzing a given text.
    @arg text: either a list of words or a string with words seperated by spaces,
               if passed empty string, returns a default dictionary

    """
    # Set up the result dictionary
    emotions = {x for y in emotion_dict.values() for x in y}
    emotion_count = dict()

    for emotion in emotions:
        emotion_count[emotion] = 0

    # If text is an empty string, return default dictionary with 0 values
    if text == "":
        return emotion_count

    # Analyze the text and normalize by total number of words
    if not isinstance(text, list):
        text = listify(text)
    num_words = len(text)
    for word in text:
        if emotion_dict.get(word): # if word exists in emotion dictionary,
            for emotion in emotion_dict.get(word): # for each emotion,
                emotion_count[emotion] += 1/num_words # update counter in results dictionary

    return emotion_count

#### TESTING AND EXAMPLES BELOW ####

# Test getting positive and negative words, and computing cross sentiment
# T = get_positive_negative_words()
# print(compute_cross_sentiment("I love you.", T[0], T[1]))

# Test getting the emotions dictionary
# E = get_emotion_dictionary()
# print(E["love"])

# Test getting valid assessment of emotions
# print(compute_emotion_sentiment("Okay. Too much snow, I am frustrated. What if I slip?", E))
