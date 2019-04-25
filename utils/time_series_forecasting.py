# Imports for historical CSV file scraping from Yahoo Finance
import re
from io import StringIO
from datetime import datetime, timedelta
import requests

# Import for time series forecasting
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Argument specification
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--download",
                    default="no",
                    help="Pass 'yes' to retrieve data from Yahoo Finance, or use in-place data.")
parser.add_argument("--load_path",
                    default="../price_data/",
                    help="Path of the loaded historical price data.")
args = parser.parse_args()

# The below class solutions is adapted from the link below:
# https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python
# Example URL for 'Download Data' in Yahoo Finance's historical data displayer:
# https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1553228964&period2=1555907364&interval=1d&events=history&crumb=4TivZ0L0oTM
# NOTE: The above link doesn't work sometimes returning 404, Client Error. Keep trying.
class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = str('https://query1.finance.yahoo.com/v7/finance/download/' +
                     '{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}')

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])

df_bitcoin = None
if args.download == 'yes': # Download from Yahoo Finance
    df_bitcoin = YahooFinanceHistory('BTC-USD', days_back=365*2).get_quote()
    # print(df_bitcoin.head()) # debugging
    df_bitcoin.to_csv(args.load_path + 'historic_bitcoin.csv', sep='\t', encoding='utf-8')
else: # load from specified data path
    df_bitcoin = pd.read_csv(args.load_path + 'historic_bitcoin.csv',
                             sep='\t', encoding='utf-8',
                             index_col=0, engine='python') # index_col = 0 removes 'Unnamed:0' column

# Convert dates to Pandas interpretable form
df_bitcoin["Date"] = pd.to_datetime(df_bitcoin["Date"], format='%Y-%m-%d')

# Drop unnecessary columns, keep only 'Open' price values
# TODO: Check if this is logical
df_bitcoin = df_bitcoin.drop(df_bitcoin.columns[2:], axis=1)

# Plot data for visualization
f, ax = plt.subplots(1)
df_bitcoin.plot(kind='line', x='Date', y='Open', title='Bitcoin Prices', ax=ax)

#### TIME-FORECASTING METHODS ####
# Prepare new data frame to-be filled with various predictions
train = np.asarray(df_bitcoin["Open"])
y_hat = df_bitcoin.copy()
n = y_hat.shape[0] - 1
# Add for ~90 days (3 months)
start_day = 1
end_day = 93
num_days = end_day - start_day
for i in range(start_day, end_day):
    new_date = y_hat.iloc[n,0]+ pd.DateOffset(days=i)
    y_hat.loc[n+i] = [new_date, 0.0]
# Drop already filled columns, keep only to be predicted rows (future)
y_hat = y_hat.drop(y_hat.index[:n+1], axis=0)

## ABSOLUTELY NAIVE APPROACH ##
# y_hat(t+1) = y(t)
y_hat["Naive"] = train[-1]
y_hat.plot(kind='line', x='Date', y='Naive', title='Bitcoin Prices', ax=ax)

## SIMPLE AVERAGE APPROACH ##
# y_hat(t+1) = SUM(y(i))/n
y_hat["SimpleAverage"] = train.mean()
y_hat.plot(kind='line', x='Date', y='SimpleAverage', title='Bitcoin Prices', ax=ax)

## MOVING AVERAGE APPROACH ##
# y_hat(t+1) = (1/k)(y(t)+y(t-1)+...+y(t-k)) based on some finite k
# TODO: Add weights
k = 50 # past observations to count as a factor
y_hat["MovingAverage"] = train[(-1)*k:].mean()
y_hat.plot(kind='line', x='Date', y='MovingAverage', title='Bitcoin Prices', ax=ax)

## SIMPLE EXPONENTIAL SMOOTHING ##
# y_hat(t+1) = a(y(t)) + a(1-a)(y(t-1)) + a(1-a)^2(y(t-2)) where 0 <= a <= 1 is the smoothing level
fit = SimpleExpSmoothing(train).fit(smoothing_level=0.6)
y_hat["SimpleExponential"] = fit.forecast(y_hat.shape[0])
y_hat.plot(kind='line', x='Date', y='SimpleExponential', title='Bitcoin Prices', ax=ax)

## HOLT'S LINEAR TREND METHOD ##
# Setup dataframe from scratch to not get np.isfinite() error
# dates = np.array(df_bitcoin["Date"], dtype=np.datetime64)
# data = np.array(df_bitcoin["Open"], dtype=np.float64)
# df_bitcoin = pd.DataFrame({'data': data}, index=dates)
# Optional: Visualize the trend
# sm.tsa.seasonal_decompose(df_bitcoin, model='additive', freq=1).plot()
# result = sm.tsa.stattools.adfuller(train)

# (1) Forecasting Equation: y(t+1) = l(t) + h*b(t)
# (2) Level Equation: l(t) = a(y(t)) + (1-a)(l(t-1)+b(t-1)) where 0 <= a <= 1 is the smoothing level
# (3) Trend Equation: b(t) = B*(l(t)-l(t-1))+(1-B)b(t-1) where 0 <= B <= 1 is the smoothing slope
fit = Holt(train).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat["Holt"] = fit.forecast(y_hat.shape[0])
y_hat.plot(kind='line', x='Date', y='Holt', title='Bitcoin Prices', ax=ax)

## HOLT-WINTER'S METHOD ##
# Only to be used when seasonality is in effect.
# QUESTION: Does seasonality effect Bitcoin prices?
# (1) Forecasting Equation: y(t+k) = L(t) + kb(t) + s(t+k-s)
# (2) Level Equation: L(t) = a(y(t)-S(t-s)) + (1-a)(L(t-1)+b(t-1))
# (3) Trend Equation: b(t) = B(L(t)-L(t-1)) + (1-B)b(t-1)
# (4) Seasonal Equation: S(t) = Y(y(t)-L(t)) + (1-Y)(S(t-s))
# where s is the length of the seasonal cycle
# 0 <= a, B, Y <= 1
fit = ExponentialSmoothing(train, seasonal_periods=7,
                           trend='add', seasonal='add').fit()
y_hat["Holt-Winter"] = fit.forecast(y_hat.shape[0])
y_hat.plot(kind='line', x='Date', y='Holt-Winter', title='Bitcoin Prices', ax=ax)

## ARIMA: AUTOREGRESSIVE INTEGRATED MOVING AVERAGE ##
#dates = np.array(df_bitcoin["Date"], dtype=np.datetime64)
#data = np.array(df_bitcoin["Open"], dtype=np.float64)
#train = pd.DataFrame({'data': data}, index=dates)
fit = sm.tsa.statespace.SARIMAX(train, order=(2,1,4),
                                seasonal_order=(0, 1, 1, 7)).fit()
y_hat["ARIMA"] = fit.predict(start=train.shape[0],
                             end=train.shape[0]+num_days-1,
                             dynamic=True)
y_hat.plot(kind='line', x='Date', y='ARIMA', title='Bitcoin Prices', ax=ax)
plt.show()
