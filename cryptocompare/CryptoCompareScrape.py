import requests
import json
import pandas as pd
import sys

#url = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key=c8df95f79affff7fbffb29cc664355497c8418f243daac00406d82be1d67bb58&ITs=1553045108'



def get_news_before(ts):
    url = 'https://min-api.cryptocompare.com/data/v2/news/?lTs={}&api_key=c8df95f79affff7fbffb29cc664355497c8418f243daac00406d82be1d67bb58'.format(ts)
    r = requests.get(url).json()
    #data = r['Data']
    df = pd.DataFrame(r['Data'])
    #for i in data:
        #print(i['published_on'])
    with open('data.csv', 'a') as f: 
    	df.to_csv(f, header=False)

epoch_date = 1540000000
while epoch_date > 1500000000:
    epoch_date_str = str(epoch_date)
    get_news_before(epoch_date_str)
    epoch_date -= 100000

#for i in data:
#	with open('data1.txt', "a") as scraped:
#		json.dump(data, scraped)
