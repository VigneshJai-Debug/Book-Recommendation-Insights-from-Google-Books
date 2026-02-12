import requests
import pandas as pd

all_items = []
url = "https://gutendex.com/books/"

while url and len(all_items) < 5000:
    response = requests.get(url).json()
    all_items.extend(response.get("results", []))
    url = response.get("next")

df = pd.json_normalize(all_items)
df = df.head(5000)
df.to_csv("gutendex-5000-books.csv", index=False)
print(df.shape)
