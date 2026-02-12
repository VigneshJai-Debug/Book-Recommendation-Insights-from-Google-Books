import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

df_original = pd.read_csv("dataset\cleaned_dataset_v3.csv")

df = df_original.copy()

text_columns = ['title', 'authors', 'summaries', 'subjects', 'bookshelves']

def clean_text(text):
    text = str(text)
    text = re.sub(r"\[|\]|'|\"", "", text)
    text = text.lower()
    return text

for col in text_columns:
    df[col] = df[col].apply(clean_text)

df['combined_text'] = df[text_columns].agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=5
)


tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

feature_names = vectorizer.get_feature_names_out()

sparse.save_npz("tfidf_matrix.npz", vectorizer.transform(df['combined_text']))


pd.DataFrame({
    "id": df["id"],
    "download_count": df["download_count"]
}).to_csv("dataset/metadata_for_ranking.csv", index=False)

np.save("tfidf_feature_names.npy", feature_names)

df.to_csv("dataset\\transformed_dataset_v1.py")

