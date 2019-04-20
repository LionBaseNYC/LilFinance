#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 06:21:56 2019
@author: uzaymacar
Code to scrape data from Reddit.
To use praw correctly, refer to the following two links:
1) https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
2) https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps
"""

def import_module(name, filepath):
    """
    Function to solve the relative import curse, used in replace of
    'from utils.sentiment_analysis_utils import *' which didn't work.

    @param name: Used to create or access a module object.
    @param filepath: Pathname argument that points to the source file.
    """
    # imp module provides an interface to the mechanisms used to implement the import statement
    import os, imp
    # @param file: The source file, open for reading as text, from the beginning.
    pathname = filepath if os.path.isabs(filepath) else os.path.join(os.path.dirname(__file__), filepath)
    return imp.load_source(name, pathname)

sentiment_analysis_utils = import_module(name = "sentiment_analysis_utils",
                                         filepath = "../utils/sentiment_analysis_utils.py")
from sentiment_analysis_utils import *
import praw # wrapper library for Reddit API
import matplotlib.pyplot as plt

# Get below information from https://www.reddit.com/prefs/apps
CLIENT_ID = 'KCuQ09njTHT2lg'
CLIENT_SECRET = 'pJ0ZLgK2hqWbgC6OLVYbju8JGAg'

# Construct user-agent string: <platform>:<app ID>:<version string> (by /u/<Reddit username>)
USER_AGENT = 'macos:KCuQ09njTHT2lg:v0.0.0 (by /u/PotentialBenefit)'

# Output file to write relevant information
log_file = open("logs/log.txt", "w")
scores_file = open("logs/crypto_scores.txt", "w")
emotions_file = open("logs/crypto_emotions.txt", "w")

# Create a Reddit instance
reddit = praw.Reddit(client_id='KCuQ09njTHT2lg',
                     client_secret='pJ0ZLgK2hqWbgC6OLVYbju8JGAg',
                     user_agent='macos:KCuQ09njTHT2lg:v0.0.0 (by /u/PotentialBenefit)')

# Specify subreddits to be scraped
SUBREDDITS = ['CryptoCurrency', 'CryptoCurrencyTrading', 'CryptoCurrencies', 'CryptoMarkets']

# Specify crytocurrencies to match
CRYPTOCURRENCIES = ['Bitcoin', 'Ethereum', 'XRP', 'EOS', 'Litecoin', 'Bitcoin Cash', 'Tether',
                    'Stellar', 'Binance Coin', 'TRON', 'Bitcoin SV', 'Cardano', 'Monero', 'IOTA',
                    'Dash', 'Maker', 'NEO', 'Ethereum Classic', 'NEM', 'Zcash']

CRYPTOCURRENCIESABBREV = [ 'BTC', 'ETH', 'XRP', 'EOS', 'LTC', 'BCH', 'USDT',
                    'BNB','TRON', 'BSV', 'ADA', 'XMR', 'MIOTA', 'DASH', 'MKR',
                     'NEO', 'ETC', 'NEM', 'ZEC']
"""
How does scoring work on Reddit submissions?
1) The Displayed Score: all votes are equal, whether earlier or later.
2) The Internal Score : votes cast later are worth less than votes cast earlier.
A post that gets lots of early upvotes that end up on the main page.
Then, the rest of Reddit sees the post on /r/all, and starts downvoting it.
Those late downvotes easily change the Displayed Score, but they don't change
the post's position on /r/all as much.
"""

SCORES = [0] * len(CRYPTOCURRENCIES)

emotion_dict = get_emotion_dictionary()
#print(emotion_dict)

#print(default_dict)
EMOTIONS = [] # do it like this to deal with instance issues of dictionaries
for i in range(len(CRYPTOCURRENCIES)):
    default_dict = compute_emotion_sentiment("", emotion_dict)
    EMOTIONS.append(default_dict)

N = 10000 # submission limit

# Get N hottest submissions from each specified subreddit
for SUBREDDIT in SUBREDDITS:
    log_file.write("SUBREDDIT: " + SUBREDDIT + "\n")
    log_file.write("-"*60 + "\n")
    hot_ID = 0
    for submission in reddit.subreddit(SUBREDDIT).hot(limit=N):
        log_file.write("ID: " + str(hot_ID) + "\n")
        log_file.write("Title: " + str(submission.title) + "\n")
        log_file.write("Score: " + str(submission.score) + "\n")
        log_file.write("URL: " + str(submission.url) + "\n")
        log_file.write("\n")
        hot_ID += 1 # update counter
        new_emotion_dict = compute_emotion_sentiment(submission.title, emotion_dict)
        for i in range(len(CRYPTOCURRENCIES)):
            if CRYPTOCURRENCIES[i] in submission.title:
                # print(CRYPTOCURRENCIES[i], submission.title)
                # print(new_emotion_dict)
                for key in EMOTIONS[i]:
                    # print(key)
                    EMOTIONS[i][key] += new_emotion_dict[key]
                SCORES[i] += submission.score
            # print(EMOTIONS)
    log_file.write("\n\n")
log_file.close()

for i in range(len(SCORES)):
    scores_file.write("CRYPTOCURRENCY: " + CRYPTOCURRENCIES[i] + "\n")
    scores_file.write("SCORE: " + str(SCORES[i]) + "\n\n")

scores_file.close()

#print(EMOTIONS)
for i in range(len(EMOTIONS)):
    emotions_file.write("CRYPTOCURRENCY: " + CRYPTOCURRENCIES[i] + "\n")
    emotions_file.write(str(EMOTIONS[i]))
    emotions_file.write("\n\n")

emotions_file.close()

print("SUCCESS!")

# Plotting preferences
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'cornsilk',
          'cadetblue', 'maroon', 'indigo', 'chocolate', 'silver',
          'blue', 'red', 'green', 'olive', 'crimson',
          'yellow', 'darkorange', 'ivory', 'olivedrab', 'slategray']

# Pie Chart for cryptocurrency scores
explode_S = (0, 0, 0, 0, 0)
plt.title("Cryptocurrency Scores")
labels = CRYPTOCURRENCIES
sizes = SCORES
plt.pie(sizes[:5], explode=explode_S, labels=labels[:5], colors=colors[:5],
            autopct='%1.1f%%', radius=5, shadow=False, startangle=140)

plt.axis('equal')
plt.savefig('figures/crpytocurrency_scores.png')
plt.close('all')

# Pie Chart for normalized emotions of each cryptocurrency
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # explode any slice you want

for i in range(len(CRYPTOCURRENCIES)):
    sizes = []
    labels = []
    for key in EMOTIONS[i]:
        labels.append(key)
        sizes.append(EMOTIONS[i][key])

    # autopct='%1.1f%%'
    plt.title(CRYPTOCURRENCIES[i])
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', radius=5, shadow=False, startangle=140)

    plt.axis('equal')
    plt.savefig('figures/' + CRYPTOCURRENCIES[i].lower() + '_normalized_emotions.png')
    plt.close('all')

# TODO: The scores gathered at this point doesn't represent a quality measure,
# rather it is just an indicator for popularity.
