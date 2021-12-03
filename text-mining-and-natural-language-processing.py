#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


# In[3]:


df = pd.read_csv('../input/df-sub/df_sub.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df['reviewText'].head(10)


# In[6]:


# Normalizing Case Folding

df['reviewText'] = df['reviewText'].str.lower()


# In[7]:


# Deleting punctuations with regex

df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')


# In[8]:


# Deleting numbers with regex

df['reviewText'] = df['reviewText'].str.replace('\d', '')


# In[9]:


# Stopwords

sw = stopwords.words('english')
print(sw)


# In[10]:


df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# In[11]:


# Rarewords

delete_ = pd.Series(' '.join(df['reviewText']).split()).value_counts()[-1000:]
delete_


# In[12]:


df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in delete_))


# In[13]:


# Tokenization

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

# no assignment, just to see


# In[14]:


df["reviewText"].head(15)


# In[15]:


# Calculating items' frequency

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.shape


# In[16]:


# Lemmatization (Finding words with the same root)

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df["reviewText"].head(15)


# In[17]:


# Calculating items' frequency

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.shape


# In[18]:


# As seen above number of words has decreased 


# # TEXT VISUALIZATION

# ## Barplot

# In[19]:


tf.columns = ["words", "tf"]
tf.head()


# In[20]:


tf[tf["tf"] > 500].plot.bar(x="words", y="tf", figsize = (20,7))
plt.show()


# ## Wordcloud

# In[21]:


text =  " ".join(i for i in df.reviewText)


# In[22]:


wordcloud = WordCloud().generate(text)


# In[23]:


plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[24]:


# some small arrangments

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[25]:


# Changing backround for word cloud
vbo_mask = np.array(Image.open("../input/pcturee/download.png"))


# In[26]:


wc = WordCloud(background_color="white",
               max_words=1000,
               mask=vbo_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[7, 7])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# # SENTIMENT ANALYSIS

# In[27]:


df.head()


# In[28]:


sia = SentimentIntensityAnalyzer()


# In[29]:


# some examples

sia.polarity_scores("The film was awesome")


# In[30]:


sia.polarity_scores("I liked this music but it is not good as the other one") 


# In[31]:


df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))


# In[32]:


df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)['compound'])


# In[33]:


df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"]


# In[34]:


df.head()


# In[35]:


df.groupby('sentiment_label').agg({'overall':'mean'})


# # Vectorization
# # 1.Count Vectors

# ### Word Frequency

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']


# In[37]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()


# In[38]:


X.toarray()


# ### n-gram Frequency

# In[39]:


vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names()


# In[40]:


X2.toarray()


# ## 2.TF-IDF
# 
# 

# In[41]:


# TF-IDF = TF(t) * IDF(t)


# ###  Word tf-idf

# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()


# In[43]:


X.toarray()


# ### n-gram tf-idf

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()


# In[45]:


X.toarray()


# # FEATURE ENGINEERING

# In[46]:


df1 = df[['reviewText','sentiment_label']]
df1.head()


# In[47]:


train_x, test_x, train_y, test_y = train_test_split(df["reviewText"],
                                                    df["sentiment_label"],
                                                    random_state=17)


# In[48]:


df.head()


# In[49]:


train_x[0:5]


# In[50]:


train_y[0:5]


# In[51]:


encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)


# In[52]:


train_y


# ##  Count Vectors

# In[53]:


vectorizer = CountVectorizer()
vectorizer.fit(train_x)


# In[54]:


x_train_count = vectorizer.transform(train_x)


# In[55]:


x_test_count = vectorizer.transform(test_x)


# In[56]:


vectorizer.get_feature_names()[0:10]


# In[57]:


x_train_count.toarray()


# ### TF-IDF Word Level

# In[58]:


tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
tf_idf_word_vectorizer.get_feature_names()[0:10]


# In[59]:


x_train_tf_idf_word.toarray()


# ### TF-IDF N-Gram Level
# 

# In[60]:


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)).fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)


# In[61]:


x_train_tf_idf_ngram.toarray()


# ### TF-IDF Characters Level

# In[62]:


tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)


# In[63]:


x_train_tf_idf_chars.toarray()


# # MODELING (SENTIMENT MODELING)
# 

# ## TF-IDF Word-Level Logistic Regression
# 

# In[64]:


log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)
y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))


# In[65]:


cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()


# ## RandomForestClassifier
# ## TF-IDF Word-Level

# In[66]:


rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()


# ### TF-IDF N-GRAM

# In[67]:


rf_model = RandomForestClassifier().fit(x_train_tf_idf_ngram, train_y)
cross_val_score(rf_model, x_test_tf_idf_ngram, test_y, cv=5, n_jobs=-1).mean()


# ### TF-IDF CHARLEVEL

# In[68]:


rf_model = RandomForestClassifier().fit(x_train_tf_idf_chars, train_y)
cross_val_score(rf_model, x_test_tf_idf_chars, test_y, cv=5, n_jobs=-1).mean()


# ### Count Vectors

# In[69]:


rf_model = RandomForestClassifier().fit(x_train_count, train_y)
cross_val_score(rf_model, x_test_count, test_y, cv=5).mean()

