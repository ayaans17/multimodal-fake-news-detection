import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("../Dataset/merged.csv")

text_column = "title"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def basic_clean(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # remove html
    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # remove mentions and hashtags if needed
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["clean_text"] = df[text_column].apply(basic_clean)