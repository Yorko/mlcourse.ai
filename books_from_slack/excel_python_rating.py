import pandas as pd
from urllib.parse import urlencode, quote_plus
import requests
base_url="https://docs.google.com/spreadsheets/d"
sheet_key="1NSdcHzMy_KoFx6hSZUuXUTwW5X6xsFY-POW2wQ4jjjs"
sheet_id="1465042576"
service="gviz/tq?"
#query="select * where A='Трунов Артем Геннадьевич'"
query="select * where A='Змеев Александр Викторович'"

payload = {'tqx':'out:csv',
           'sheet':sheet_id,
           'tq':query}
encoded = urlencode(payload, quote_via=quote_plus)
sheet_url = "/".join([base_url,sheet_key,service]) + encoded

#print (sheet_url)
r = requests.get(url=sheet_url)
print(r.text)

from io import StringIO
pd.read_csv(StringIO(r.text))