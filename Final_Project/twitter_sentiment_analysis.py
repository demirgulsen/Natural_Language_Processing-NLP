################################################
# Makine Öğrenmesi ile Twitter Duygu  Analizi
################################################
# İş Problemi
# 2022 yılına ait Türkçe tweetler çekilerek Twitter kullanıcıları tarafından atılan tweetlerin taşıdıkları duygu kapsamında pozitif, negatif ve nötr olarak tahmin edilmesi
################################################
# tweets_labeled : 2022 yılına ait tweet bilgilerini,
# tweets_21 : 2021 yılına ait tweetleri içermektedir

################################################
# Feature Engineering
################################################
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
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

data_labeled = pd.read_csv('Github_Natural_Language_Processing(NLP)/Final_Project/datasets/tweets_labeled.csv')
data_21 = pd.read_csv('Github_Natural_Language_Processing(NLP)/Final_Project/datasets/tweets_21.csv')

def data_info(dataframe):
    print("###### HEAD ############")
    print(dataframe.head())
    print("###### SHAPE ############")
    print(dataframe.shape)
    print("###### INFO ############")
    print(dataframe.info())
    print("###### NULL COUNT ############")
    print(dataframe.isnull().sum())

data_info(data_labeled)
data_info(data_21)

df = data_labeled.copy()
# Eksik Değer Analizi
# tweets_labeled veri setinde sadece 1 tane boş değer var onu da silelim
df = df.dropna()

# object tipindeki date değişkenini datetime tipine çevirelim
df['date'] = pd.to_datetime(df['date'])

# Zaman dilimini GMT+03:00 olarak ayarlayalım
df["date"]= df["date"].dt.tz_convert("Etc/GMT-3")

# date değişkenini kullanarak yeni değişkenler üretelim
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour

# Mevsim değişkeni oluşturalım
bins = [1, 3, 6, 9, 12]  # Kış (1-3), İlkbahar (4-6), Yaz (7-9), Sonbahar (10-12)
labels = ['winter', 'spring', 'summer', 'autumn']
df['season'] = pd.cut(df['month'], bins=bins, labels=labels, right=True)


# 4 saatlik aralıkları belirle
bins = [0, 4, 8, 12, 16, 20, 24]
labels = ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-00:00']
df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

# Sonuçları kontrol et
df[['date', 'hour', 'time_period']].head()


df['label_str'] = df['label'].astype(int).replace({1: 'pozitif', 0: 'neutral', -1: 'negatif'})


grouped_df = df.groupby('label_str').size()
# Pasta grafiği çizimi
df.groupby('label_str').size().plot(kind='pie', autopct='%1.1f%%')
# Grafiği gösterme
plt.axis('equal')  # Daire şeklinde görüntülemek için
plt.show()

#############################################################
# Numerik ve kategorik değişkenleri yakalayalım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "tweet_id"]


# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

############################
 # Sayısal Değişken Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.figure(figsize=(8, 5))
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)


# Hedef değişken analizi yapalım. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

# Kategorik değişkenlere göre hedef değişkenin analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Count": dataframe.groupby(categorical_col)[target].count(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"label", col)


# hedef değişkene göre numerik değişkenlerin analizi
def target_summary_with_num(dataframe, target, num_col):
    print(pd.DataFrame({num_col + "_MEAN": dataframe.groupby(target)[num_col].mean()}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df,"label", col)


################################################
# Veri Hazırlama ve Modelleme
################################################
df['tweet'] = df['tweet'].str.lower()

df['tweet'] = df['tweet'].str.replace('[^\w\s]', '', regex=True)

df['tweet'] = df['tweet'].str.replace('\n', '', regex=True)

df['tweet'] = df['tweet'].str.replace('\d', '', regex=True)

# link içerip içermediğini kontrol edelim
# df['tweet'].str.contains(r"(https?://\S+|www\.\S+|\S+\.\S+)", regex=True)
df['tweet'] = df['tweet'].str.replace(r"(https?://\S+|www\.\S+|\S+\.\S+)", "", regex=True)

# Stopwords
sw = stopwords.words('turkish')
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords - Nadir Kelimeler
temp_df = pd.Series(' '.join(df['tweet']).split()).value_counts()[-50:]
drops = temp_df[temp_df == 1]
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization  - Kelimeleri Köklerine Ayırır
df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################
# Tekrarlayan kelimeleri kaldıran fonksiyon
def remove_duplicate_words(text):
    words = text.split()
    unique_words = list(dict.fromkeys(words))  # Sıralı olarak benzersiz kelimeleri al
    return " ".join(unique_words)

# Seride uygula
df['tweet'] = df['tweet'].apply(remove_duplicate_words)

##################

# Terim Frekanslarının Hesaplanması
tf = df["tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Barplot
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()


# Wordcloud
text = " ".join(i for i in df.tweet)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#################################################################
# MODELLING

train_x, test_x, train_y, test_y = train_test_split(df["tweet"],
                                                    df["label"],
                                                    test_size=0.3,
                                                    random_state=42)

# TF-IDF
tf_idf_word_vectorizer = TfidfVectorizer()
x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Lojistik Regresyon
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
cross_val_score(log_model, x_test_tf_idf_word, test_y, scoring="accuracy", cv=5).mean()

#######################################
# Hiperparametre optimizasyonu
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],  # Regularization parametresi
    "max_iter": [100, 200, 500, 1000]  # Maksimum iterasyon sayısı
}

grid_search = GridSearchCV(log_model, param_grid, cv=10, scoring="accuracy", n_jobs=4, verbose=1)
grid_search.fit(x_train_tf_idf_word, train_y)

# En iyi parametreler
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk:", grid_search.best_score_)
cross_val_score(grid_search, x_test_tf_idf_word, test_y, scoring="accuracy", cv=10).mean()

################################################
# Tweetlerde Duygu Tahmini
################################################
# tweets_21 veri setine ait duygu tahmini işlemini gerçekleştirelim


data_21.head()
data_21.info()

data_21["date"] = pd.to_datetime(data_21["date"])  # Tarih formatına çevir
data_21["date"] = data_21["date"].dt.tz_localize("UTC").dt.tz_convert("Etc/GMT-3")

data_21['tweet'] = data_21['tweet'].str.lower()
# data_21['tweet'] = data_21['tweet'].str.replace('[^\w\s]', '', regex=True)
# data_21['tweet'] = data_21['tweet'].str.replace('\n', '', regex=True)
# data_21['tweet'] = data_21['tweet'].str.replace('\d', '', regex=True)



# Lojistik Regresyon

random_review = pd.Series(data_21["tweet"].sample(1).values)
new_review = tf_idf_word_vectorizer.transform(random_review)
sentiment_prediction = log_model.predict(new_review)
data_21['label'] = sentiment_prediction

