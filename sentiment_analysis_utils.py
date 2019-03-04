#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 4 02:19:56 2019
@author: uzaymacar
Utilities included for performing sentiment analysis, adapted from
LionBase's Sentiment Analysis workshop.
"""

import requests

def listify(text):
    """
    Function convert a string text to a list of words.
    @arg text: space seperated words
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
    sentiment analysis.
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
    emotions_file = "sentiment_data/NRC-emotion-lexicon-wordlevel-alphabetized.txt"
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
    @arg text: either a list of words or a string with words seperated by spaces
    """
    # Set up the result dictionary
    emotions = {x for y in emotion_dict.values() for x in y}
    emotion_count = dict()
    for emotion in emotions:
        emotion_count[emotion] = 0
    
    # Analyze the text and normalize by total number of words
    if not isinstance(text, list):
        text = listify(text)
    num_words = len(text)
    for word in text:
        if emotion_dict.get(word): # if word exists in emotion dictionary,
            for emotion in emotion_dict.get(word): # for each emotion,
                emotion_count[emotion] += 1/num_words # update counter in results dictionary

    return emotion_count

# TESTING AND EXAMPLES BELOW

# Test getting positive and negative words, and computing cross sentiment
T = get_positive_negative_words()
print(compute_cross_sentiment("I love you.", T[0], T[1]))

# Test getting the emotions dictionary
E = get_emotion_dictionary()
print(E["love"])

# Test getting valid assessment of emotions
print(compute_emotion_sentiment("Okay. Too much snow, I am frustrated. What if I slip?", E))
