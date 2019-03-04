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

import praw # wrapper library for Reddit API

# Get below information from https://www.reddit.com/prefs/apps
CLIENT_ID = 'KCuQ09njTHT2lg'
CLIENT_SECRET = 'pJ0ZLgK2hqWbgC6OLVYbju8JGAg'
# Construct user-agent string: <platform>:<app ID>:<version string> (by /u/<Reddit username>)
USER_AGENT = 'macos:KCuQ09njTHT2lg:v0.0.0 (by /u/PotentialBenefit)'

# Output file to write relevant information
log_file = open("log.txt", "w")
scores_file = open("crypto_scores.txt", "w")

# Create a Reddit instance
reddit = praw.Reddit(client_id='KCuQ09njTHT2lg',
                     client_secret='pJ0ZLgK2hqWbgC6OLVYbju8JGAg',
                     user_agent='macos:KCuQ09njTHT2lg:v0.0.0 (by /u/PotentialBenefit)')

SUBREDDITS = ['CryptoCurrency', 'CryptoCurrencyTrading', 'CryptoCurrencies', 'CryptoMarkets']

CRYPTOCURRENCIES = ['Bitcoin', 'Ethereum', 'XRP', 'EOS', 'Litecoin', 'Bitcoin Cash', 'Tether',
                    'Stellar', 'Binance Coin', 'TRON', 'Bitcoin SV', 'Cardano', 'Monero', 'IOTA',
                    'Dash', 'Maker', 'NEO', 'Ethereum Classic', 'NEM', 'Zcash']

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

N = 100 # submission limit

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
        for i in range(len(CRYPTOCURRENCIES)):
            if CRYPTOCURRENCIES[i] in submission.title:
                SCORES[i] += submission.score
    log_file.write("\n\n")

log_file.close()

for i in range(len(SCORES)):
    scores_file.write("CRYPTOCURRENCY: " + CRYPTOCURRENCIES[i] + "\n")
    scores_file.write("SCORE: " + str(SCORES[i]) + "\n")

scores_file.close()
print("SUCESS!")

# TODO: The scores gathered at this point doesn't represent a quality measure,
# rather it is just an indicator for popularity.
