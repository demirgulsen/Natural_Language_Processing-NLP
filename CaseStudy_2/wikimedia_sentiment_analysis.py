#################################################
# WIKIPEDIA - Metin Ön işleme ve Görselleştirme (NLP - Text Preprocessing & Text Visualization)
#################################################

############
# Problem
############
# Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapınız.


###################
# Proje Görevleri
###################

#################################################
# Görev 1: Metin Ön İşleme İşlemlerini Gerçekleştiriniz
#################################################
# Adım1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
# • Büyük küçük harf dönüşümü,
# • Noktalama işaretlerini çıkarma, pw
# • Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

df = pd.read_csv('Github_Natural_Language_Processing(NLP)/CaseStudy_2/dataset/wiki_data.csv', index_col=0)

df.head()
df.shape
df.isnull().sum()

def clean_text(text):
    text = text.str.lower()
    text = text.str.replace('[^\w\s]', '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    text = text.str.replace('\n', ' ')

    return text


# Adım2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.
df['text'] = clean_text(df['text'])

df.head()

# Adım3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon yazınız.
def remove_stopwords(text):
    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return text

# Adım4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.
df['text'] = remove_stopwords(df['text'])

df.head()

# Adım5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.
temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
drops = temp_df[temp_df < 1000]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


# Adım6: Metinleri tokenize edip sonuçları gözlemleyiniz.
df["text"].apply(lambda x: TextBlob(x).words).head()

# Adım7: Lemmatization işlemi yapınız.
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#################################################
# Görev 2: Veriyi Görselleştiriniz
#################################################
# Adım1: Metindeki terimlerin frekanslarını hesaplayınız.
tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Adım2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.
tf[tf["tf"] > 10000].plot.bar(x="words", y="tf")[50]
plt.show()

# Adım3: Kelimeleri WordCloud ile görselleştiriniz.
text = " ".join(i for i in df.text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#################################################
# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
#################################################
# Adım1: Metin ön işleme işlemlerini gerçekleştiriniz.
# Adım2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# Adım3: Fonksiyonu açıklayan 'docstring' yazınız.

def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
        Textler Ã¼zerinde Ã¶n iÅŸleme iÅŸlemleri yapar.

        :param text: DataFrame'deki textlerin olduÄŸu deÄŸiÅŸken
        :param Barplot: Barplot gÃ¶rselleÅŸtirme
        :param Wordcloud: Wordcloud gÃ¶rselleÅŸtirme
        :return: text


        Example:
                wiki_preprocess(dataframe[col_name])
        """

    text = text.str.lower()
    text = text.str.replace('[^\w\s]', '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    text = text.str.replace("\n\n", " ")
    # Stopwords
    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    temp_df = pd.Series(' '.join(text).split()).value_counts()
    drops = temp_df[temp_df < 1000]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    # Lemmatization
    text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    if Barplot:
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > 20000].plot.bar(x="words", y="tf")[50]
        plt.show()

    if Wordcloud:
        text = " ".join(i for i in df.text)
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text


df = pd.read_csv('Github_Natural_Language_Processing(NLP)/CaseStudy_2/dataset/wiki_data.csv', index_col=0)

df["text"] = wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)

df.head()


# Modelling
def model_pred(df, random_review):
    sia = SentimentIntensityAnalyzer()
    df["sentiment_label"] = df['text'].apply(lambda x: 1 if sia.polarity_scores(x)["compound"] > 0 else 0)

    train_x, test_x, train_y, test_y = train_test_split(df["text"],
                                                        df["sentiment_label"],
                                                        random_state=42)
    tf_idf_word_vectorizer = TfidfVectorizer()
    x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
    x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

    log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
    y_pred = log_model.predict(x_test_tf_idf_word)
    print(classification_report(y_pred, test_y))

    print('Logistic Regression Model Score: ', cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean())

    # Prediction
    new_review = CountVectorizer().fit(train_x).transform(random_review)
    pred = log_model.predict(new_review)
    print(f'Review:  {random_review[0]} \n Prediction: {pred}')

    rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
    print('Random Forest Model Score: ', cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean())


random_review = pd.Series(df["text"].sample(1).values)
model_pred(df, random_review)