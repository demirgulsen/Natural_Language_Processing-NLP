##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

##################################################
# Business Problem
##################################################
# Amazon üzerinden satışlarını gerçekleştiren ev tesktili ve günlük giyim odaklı üretimler yapan Kozmos ürünlerine
# gelen yorumları analiz ederek ve aldığı şikayetlere göre özelliklerini geliştirerek satışlarını artırmayı hedeflemektedir.
# Bu hedef doğrultusunda yorumlara duygu analizi yapılarak etiketlencek ve   etiketlenen veri ile sınıflandırma modeli
# oluşturulacaktır.

##################################################
# Veri Seti Hikayesi
##################################################
# Veri seti belirli bir ürün grubuna ait yapılan yorumları, yorum başlığını, yıldız sayısını ve yapılan yorumu
# kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

# Review: Ürüne yapılan yorum
# Title: Yorum içeriğine verilen başlık, kısa yorum
# HelpFul: Yorumu faydalı bulan kişi sayısı
# Star: Ürüne verilen yıldız sayısı

##########################################
# Görevler
##########################################
##########################################
# Görev 1: Metin ön işleme işlemleri.
##########################################
# 1. amazon.xlsx datasını okutunuz.
# 2. "Review" değişkeni üzerinde
    # a. Tüm harfleri küçük harfe çeviriniz
    # b. Noktalama işaretlerini çıkarınız
    # c. Yorumlarda bulunan sayısal ifadeleri çıkarınız
    # d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız
    # e. 1000'den az geçen kelimeleri veriden çıkarınız
    # f. Lemmatization işlemini uygulayınız

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from PIL import Image
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer
filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


df = pd.read_excel('Github_Natural_Language_Processing(NLP)/CaseStudy_1/dataset/amazon.xlsx')

df.head()
df.shape
df.isnull().sum()
df = df.dropna()

# Tüm harfleri küçük harfe çevirelim
df['Review'] = df['Review'].str.lower()

# Noktalama işaretlerini çıkaralım
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Yorumlarda bulunan sayısal ifadeleri çıkaralım
df['Review'] = df['Review'].str.replace('/d', '', regex=True)

# Bilgi içermeyen kelimeleri (stopwords) veriden çıkaralım
sw = stopwords.words('english')
df['reviewText'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# 1000'den az geçen kelimeleri veriden çıkaralım
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df < 1000]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization işlemini uygulayalım
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################
# Görev 2: Metin Görselleştirme
##################################

# Adım 1: Barplot görselleştirme işlemi
    # a. "Review" değişkeninin içerdiği kelimeleri frekanslarını hesaplayınız, tf olarak kaydediniz
    # b. tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
    # c. "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini tamamlayınız.

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Adım 2: WordCloud görselleştirme işlemi
    # a. "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz
    # b. WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz
    # c. Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
    # d. Görselleştirme adımlarını tamamlayınız. (figure, imshow, axis, show)

text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("Github_Natural_Language_Processing(NLP)/wordcloud.png")

##################################
# Görev 3: Duygu Analizi
##################################
# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz
sia = SentimentIntensityAnalyzer()

# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarının incelenmesi
    # a. "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız
    # b. İncelenen ilk 10 gözlem için compund skorlarına göre filtrelenerek tekrar gözlemleyiniz
    # c. 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz
    # d. "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e ekleyiniz

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])


df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])


df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# NOT:SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken oluşturulmuş oldu.

##########################################
# Görev 4: Makine öğrenmesine hazırlık!
##########################################
# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olara ayırınız.
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte.
    # a. TfidfVectorizer kullanarak bir nesne oluşturunuz.
    # b. Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
    # c. Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.


tf_idf_word_vectorizer = TfidfVectorizer()
x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

##########################################
# Görev 5: Modelleme (Lojistik Regresyon)
##########################################
# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz.
    # a. Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
    # b. classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
    # c. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız

y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

# Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması.
    # a. sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçierek yeni bir değere atayınız
    # b. Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
    # c. Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
    # d. Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
    # e. Örneklemi ve tahmin sonucunu ekrana yazdırınız.

random_review = pd.Series(df["reviewText"].sample(1).values)

new_review = CountVectorizer().fit(train_x).transform(random_review)

pred = log_model.predict(new_review)

print(f'Review:  {random_review[0]} \n Prediction: {pred}')

##########################################
# Görev 6: Modelleme (Random Forest)
##########################################
# Adım 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
    # a. RandomForestClassifier modelini kurup fit ediniz.
    # b. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız
    # c. Lojistik regresyon modeli ile sonuçları karşılaştırınız.

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)

cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# Lojistik regresyon modeli ile sonuçları karşılaştıralım
cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()
# 0.95099

cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
# 0.95389