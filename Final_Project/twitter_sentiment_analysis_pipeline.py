from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def data_prep(dataframe):
    # tweets_labeled veri setinde sadece 1 tane boş değer var onu da silelim
    dataframe = dataframe.dropna()

    # object tipindeki date değişkenini datetime tipine çevirelim
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    # Zaman dilimini GMT+03:00 olarak ayarlayalım
    dataframe["date"] = dataframe["date"].dt.tz_convert("Etc/GMT-3")

    # date değişkenini kullanarak yeni değişkenler üretelim
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['day'] = dataframe['date'].dt.day_name()
    dataframe['hour'] = dataframe['date'].dt.hour

    # Mevsim değişkeni oluşturalım
    bins = [1, 3, 6, 9, 12]  # Kış (1-3), İlkbahar (4-6), Yaz (7-9), Sonbahar (10-12)
    labels = ['winter', 'spring', 'summer', 'autumn']
    dataframe['season'] = pd.cut(dataframe['month'], bins=bins, labels=labels, right=True)

    # 4 saatlik aralıkları belirle
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-00:00']
    dataframe['time_period'] = pd.cut(dataframe['hour'], bins=bins, labels=labels, right=False)

    dataframe['label_str'] = dataframe['label'].astype(int).replace({1: 'pozitif', 0: 'neutral', -1: 'negatif'})
    dataframe['tweet'] = dataframe['tweet'].str.lower()
    dataframe['tweet'] = dataframe['tweet'].str.replace('[^\w\s]', '', regex=True)
    dataframe['tweet'] = dataframe['tweet'].str.replace('\n', '', regex=True)
    dataframe['tweet'] = dataframe['tweet'].str.replace('\d', '', regex=True)

    # Stopwords
    sw = stopwords.words('turkish')
    dataframe['tweet'] = dataframe['tweet'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # Rarewords - Nadir Kelimeler
    temp_df = pd.Series(' '.join(dataframe['tweet']).split()).value_counts()[-50:]
    drops = temp_df[temp_df == 1]
    dataframe['tweet'] = dataframe['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    # Lemmatization  - Kelimeleri Köklerine Ayırır
    dataframe['tweet'] = dataframe['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # # Terim Frekanslarının Hesaplanması
    # tf = dataframe["tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    # tf.columns = ["words", "tf"]

    dataframe["label"] = LabelEncoder().fit_transform(dataframe["label"])
    # dataframe["label"] = LabelEncoder().fit_transform(dataframe["label_str"])
    return dataframe

def logistic_regression(dataframe):
    train_x, test_x, train_y, test_y = train_test_split(dataframe["tweet"],
                                                        dataframe["label"],
                                                        random_state=42)

    # TF-IDF
    tf_idf_word_vectorizer = TfidfVectorizer()
    x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
    x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

    # Lojistik Regresyon
    log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
    print('LR - accuracy: ', cross_val_score(log_model, x_test_tf_idf_word, test_y, scoring="accuracy", cv=5).mean())

    return tf_idf_word_vectorizer, log_model


def sentiment_prediction(dataframe, tf_idf_word_vectorizer, log_model, random_review):

    new_review = tf_idf_word_vectorizer.transform(random_review)
    sentiment_prediction = log_model.predict(new_review)
    dataframe['prediction'] = sentiment_prediction

    return dataframe


def main():
    data_labeled = pd.read_csv('Github_Natural_Language_Processing(NLP)/Final_Project/datasets/tweets_labeled.csv')
    data_21 = pd.read_csv('Github_Natural_Language_Processing(NLP)/Final_Project/datasets/tweets_21.csv')

    df = data_prep(data_labeled)

    tf_idf_word_vectorizer, log_model = logistic_regression(df)

    random_review = pd.Series(df["reviewText"].sample(1).values)
    new_df = sentiment_prediction(df, tf_idf_word_vectorizer, log_model, random_review)


if __name__ == "__main__":
    print("The process has started.")
    main()
