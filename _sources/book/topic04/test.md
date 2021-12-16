---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(test)=


# Test


```{figure} /_static/img/ods_stickers.jpg
```

```{code-cell} ipython3
import os

def download_file_from_gdrive(file_url, filename, out_path='../../_static'):
    os.system(f'gdown {file_url} -O {out_path}/{filename}')
    
```


```{code-cell} ipython3
# some necessary imports
from pathlib import Path
import pandas as pd

FILE_URL = 'https://drive.google.com/uc?id=1KbBdJaEY8RF4GXzoihWgH0RdqoVBZ_oi'
FILE_NAME = 'train-balanced-sarcasm.csv.zip'
DATA_PATH = '../../_static/data/'

download_file_from_gdrive(file_url=FILE_URL, filename= FILE_NAME, out_path=DATA_PATH)
```

```{code-cell} ipython3

train_df = pd.read_csv(DATA_PATH + FILE_NAME)
train_df.head()
```

