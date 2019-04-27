''' Create a simple stocks correlation dashboard.

Choose stocks to compare in the drop down widgets, and make selections
on the plots to update the summary and histograms accordingly.

.. note::
    Running this example requires downloading sample data. See
    the included `README`_ for more information.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve stocks

at your command prompt. Then navigate to the URL

    http://localhost:5006/stocks

.. _README: https://github.com/bokeh/bokeh/blob/master/examples/app/stocks/README.md

'''
try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print ("WARNING: Cache for this example is available on Python 3 only.")
    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

from os.path import dirname, join

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure

DATA_DIR = "../price_data"

DEFAULT_TICKERS = ['BTC', 'ETH']

MAPPING = {'BTC':'bitcoin',
           'ETH':'ethereum'}

@lru_cache()
def load_ticker(ticker):
    fname = join(DATA_DIR, 'historic_%s.csv' % MAPPING[ticker])
    # data = pd.read_csv(fname, header=None, parse_dates=['date'],
                       # names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    # data = data.set_index('date')
    data = pd.read_csv(fname,
                       sep='\t', encoding='utf-8',
                       index_col=0, engine='python') # index_col = 0 removes 'Unnamed:0' column

    # Convert dates to Pandas interpretable form
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
    data = data.set_index("Date")
    return pd.DataFrame({ticker: data.Close, ticker+'_returns': data.Close.diff()})

@lru_cache()
def get_data(t1):
    df1 = load_ticker(t1)
    data = df1
    data = data.dropna()
    data['t1'] = data[t1]
    data['t1_returns'] = data[t1+'_returns']
    return data

ticker1 = Select(value='BTC', options=DEFAULT_TICKERS)

# set up widgets
source = ColumnDataSource(data=dict(date=[], t1=[], t1_returns=[]))
source_static = ColumnDataSource(data=dict(date=[], t1=[], t1_returns=[]))
tools = 'pan,wheel_zoom,xbox_select,reset'

# set up plots
ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1.line('Date', 't1', source=source_static)
ts1.circle('Date', 't1', size=1, source=source, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2.x_range = ts1.x_range
ts2.line('Date', 't1', source=source_static)
ts2.circle('Date', 't1', size=1, source=source, color=None, selection_color="orange")

# set up callbacks

def ticker1_change(attrname, old, new):
    update()

def update(selected=None):
    t1 = ticker1.value

    data = get_data(t1)
    source.data = source.from_df(data[['t1', 't1_returns']])
    source_static.data = source.data

    ts1.title.text, ts2.title.text = t1, t1

ticker1.on_change('value', ticker1_change)

def selection_change(attrname, old, new):
    t1, t2 = ticker1.value, ticker2.value
    data = get_data(t1)
    selected = source.selected.indices
    if selected:
        data = data.iloc[selected, :]

source.selected.on_change('indices', selection_change)

# set up layout
widgets = column(ticker1)
main_row = row(widgets)
series = column(ts1, ts2)
layout = column(main_row, series)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "Stocks"
