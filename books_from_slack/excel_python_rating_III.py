import pandas as pd
from urllib.parse import urlencode, quote_plus
import requests
base_url="https://docs.google.com/spreadsheets/d"
sheet_key="1HMn7uA8CwfBoWuDkiJkVomkKkoOVXyIm-vKFZlISotk"
sheet_id="2026966465"
service="gviz/tq?"
#query="select * where A='Трунов Артем Геннадьевич'"
#query="select * where A='Змеев Александр Викторович'"
query="select * where A='Топорнин Дмитрий Дмитриевич'"
payload = {'tqx':'out:csv',
           'sheet':sheet_id,
           'tq':query}
encoded = urlencode(payload, quote_via=quote_plus)
sheet_url = "/".join([base_url,sheet_key,service]) + encoded

print (sheet_url)
r = requests.get(url=sheet_url)
print(r.text)

#from io import StringIO
#pd.read_csv(StringIO(r.text))