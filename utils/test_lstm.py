# Standard imports
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
# For predictions (sentiment classification)
from sentiment_analysis_utils import sample_predict

# Progress bar for loading
def progressBar(value, endvalue, report, title="Default Process", bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0}, Percent Processed: [{1}] {2}%, Moving Average: {3}".
                     format(title, arrow + spaces, int(round(percent * 100)), report))
    sys.stdout.flush()

# Argument specification
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load_folder",
                    default="../saved_models/",
                    help="Name of the directory of the to-be loaded model file")
parser.add_argument("--load_model_name",
                    default="model",
                    help="Name of the to-be loaded model file")
args = parser.parse_args()

# Prepare (initialize) the model and required parameters
import train_lstm
# Acquire prepared model class object and the parallel vocab-to-int mapping dict
model = train_lstm.model
vocab_to_int = train_lstm.vocab_to_int
# This initializes the variables used by the optimizers, as well as any stateful metric variables
model.train_on_batch(x=np.array(train_lstm.X_train[:1], dtype=np.int32),
                     y=np.array(train_lstm.Y_train[:1], dtype=np.int32))
# Load previously trained Keras model
model.load_weights(args.load_folder + args.load_model_name)
print("Loaded model from disk...")

# Specify crytocurrencies to match
"""
# Thinking: Although articles are categorized, in the end we care more if the actual keyword
# (Bitcoin, Ethereum, etc.) is textually included in the article or not...
CATEGORIES = ['BTC', 'ETH', 'XRP', 'EOS', 'LTC', 'BCH', 'USDT',
              'BNB','TRON', 'BSV', 'ADA', 'XMR', 'MIOTA', 'DASH', 'MKR',
              'NEO', 'ETC', 'NEM', 'ZEC']
"""
CRYPTOCURRENCIES = ['Bitcoin', 'Ethereum', 'XRP', 'EOS', 'Litecoin', 'Bitcoin Cash', 'Tether',
                    'Stellar', 'Binance Coin', 'TRON', 'Bitcoin SV', 'Cardano', 'Monero', 'IOTA',
                    'Dash', 'Maker', 'NEO', 'Ethereum Classic', 'NEM', 'Zcash']
crypto_dict = {curr:[] for curr in CRYPTOCURRENCIES}
crypto_sentiment_scores = {curr:0 for curr in CRYPTOCURRENCIES}

#### DATA READING AND CATEGORIZING ####
texts = []
corresponding_categories = []
df = pd.read_csv('../cryptocompare/data.csv')
for index, row in df.iterrows():
    text, categories = row[1], row[2]
    # corresponding_categories.append(row[2])
    for curr in CRYPTOCURRENCIES:
        if curr.lower() in str(text).lower():
            crypto_dict[curr].append(text)

# Predict on a sample text without padding
sample_pred_text = "I don't really trust DogeCoin. "
sample_pred_text += "The name is cute, but is doomed to be replaced by another cryptocurrency!"
predictions = sample_predict(model,
                             sample_pred_text,
                             vocab_to_int,
                             model.seq_length)

print(predictions) # < 0.5 : Negative sentiment, >= 0.5 : Positive sentiment

#### AVERAGE SENTIMENT ASSIGNING ####
for curr in crypto_dict.keys():
    total_score = 0
    total_entrys = len(crypto_dict[curr])
    task = 0
    progressBar(value=task, endvalue=total_entrys, report=0, title=curr+" Articles")
    for article in crypto_dict[curr]:
        index = article.find(curr)
        total_score += sample_predict(model=model, X=article,
                                      vocabulary_to_integer=vocab_to_int,
                                      sequence_length=model.seq_length)
        task += 1
        progressBar(value=task, endvalue=total_entrys,
                    report=str(round(float(total_score/task), 2)),
                    title=curr+" Articles")
    crypto_sentiment_scores[curr] = total_score / total_entrys
    print("Finished " + curr + " overall sentiment assignments!")

print(crypto_sentiment_scores)
