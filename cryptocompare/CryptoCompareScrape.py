#Gets latest news articles from cryptocompare API and dumps into txt file with json format.
#Article body is under the key "body"

import requests
import json


url = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN'

r = requests.get(url).json()
data = r['Data']
for i in data:
	with open('data.txt', "a") as scraped:
		json.dump(data, scraped)
