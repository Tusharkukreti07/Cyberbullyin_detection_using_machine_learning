#!/usr/bin/env python
# coding: utf-8

# In[138]:


import nltk
import string
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from wordcloud import WordCloud


# #***1. Import File***

# In[41]:


import pandas as pd

# Specify the file path
file = r'C:\Users\owner\Downloads\kiwi\CyberBullyingTypesDataset.csv'

# Create a DataFrame from the list of lines
#df = pd.read_csv(file,header=None,names = ['comments','category'])
df = pd.read_csv(file)

#To change types to numerics
df['category'] = df['category'].replace({'Sexual Harassment': 1, 'Cyberstalking': 2, 'Doxing': 3, 'Revenge Porn': 4, 'Slut Shaming': 5})

# Display the DataFrame
df


# #***2. Data Preprocessing***

# In[42]:


spell = SpellChecker(language='en')
ps = PorterStemmer()
def transform_text(text):

    #To lower
    text = text.lower()

    #Remove Links
    pattern = r'\b(?:https?://|www\.)\S+\b'
    text = re.sub(pattern, '', text)

    #Tokenize
    text = nltk.word_tokenize(text)
    
    #Spelling Correction
    words = text[:]
    misspelled = spell.unknown(words)
    corrected_text = []
    # Correct misspelled words, excluding words with digits
    for word in words:
        if word.isnumeric() or word.isalnum() or word not in misspelled:
            corrected_text.append(word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(word)
                    
    text = corrected_text[:]
    

    #Remove special characters
    temp = []
    for i in text:
        if i.isalnum():
            temp.append(i)

    text = temp[:]
    temp.clear()

    #Remove stopwords and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)

    #Stemming
    text = temp[:]
    temp.clear()

    for i in text:
        temp.append(ps.stem(i))

    text.clear()
    
    # Remove empty strings
    for word in temp:
        if word != "" and word != "\n":
            text.append(word)
                    
    return ' '.join(text)


# In[43]:


df['processed'] = df['comments'].apply(transform_text)
df.head()


# #***3. Missing and Duplicate***

# In[44]:


df['processed'].isnull().sum()
#cpy = df['processed']


# In[45]:


df['processed'].duplicated().sum()


# In[46]:


df.drop_duplicates(subset='processed', inplace=True)


# In[47]:


df['processed'].duplicated().sum()


# In[48]:


df = df.reset_index(drop=True)


# In[49]:


df.info()


# In[50]:


df['processed'].isnull().sum()


# In[56]:


# Remove empty strings and drop NaN
# Replace empty strings and "\n" with NaN
df['processed'].replace(["", "\n"], np.nan, inplace=True)
df.dropna(subset='processed', inplace=True)


# In[58]:


df['processed'].isnull().sum()
df.info()


# #***4. Exploratory Data Analysis (EDA)***

# In[59]:


#Number of characters
df['num_char'] = df['processed'].apply(len)
df.head()


# In[60]:


#Number of words
df['num_word'] = df['processed'].apply(lambda x : len(nltk.word_tokenize(x)))
df.head()


# In[61]:


df[['num_char','num_word']].describe()


# In[62]:


df.head()


# In[64]:


#df[df['category']==0][['num_char','num_word']].describe()


# In[65]:


df[df['category']==1][['num_char','num_word']].describe()


# In[66]:


df[df['category']==2][['num_char','num_word']].describe()


# In[67]:


df[df['category']==3][['num_char','num_word']].describe()


# In[68]:


df[df['category']==4][['num_char','num_word']].describe()


# In[69]:


df[df['category']==5][['num_char','num_word']].describe()


# #***5. Word Cloud***

# In[91]:


wc = WordCloud(width=300,height=300,min_font_size=10,background_color='White')


# In[ ]:


'''notcb = wc.generate(df[df['category']==]['processed'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(notcb)'''


# In[97]:


age = wc.generate(df[df['category']==1]['processed'].str.cat(sep=" "))
plt.figure(figsize=(6,6))
plt.imshow(age)


# In[99]:


eth = wc.generate(df[df['category']==2]['processed'].str.cat(sep=" "))
plt.figure(figsize=(4,4))
plt.imshow(eth)


# In[101]:


gen = wc.generate(df[df['category']==3]['processed'].str.cat(sep=" "))
plt.figure(figsize=(6,6))
plt.imshow(gen)


# In[103]:


oth = wc.generate(df[df['category']==4]['processed'].str.cat(sep=" "))
plt.figure(figsize=(3,3))
plt.imshow(oth)


# In[105]:


rel = wc.generate(df[df['category']==5]['processed'].str.cat(sep=" "))
plt.figure(figsize=(6,6))
plt.imshow(rel)


# In[106]:


df.head()


# #***6. Model Building***

# In[107]:


# First use CountVectorizer then TFIDF to compare and choose the one with better results
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()


# In[127]:


#1. has better result in cyber bully types dataset
X = cv.fit_transform(df['processed']).toarray()
#2. X = tfidf.fit_transform(df['processed']).toarray()
X.shape


# In[128]:


y = df['category'].values
y


# In[129]:


from sklearn.model_selection import train_test_split


# In[130]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[131]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[132]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[133]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1, average='micro'))


# In[134]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2, average='micro'))


# In[135]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3, average='micro'))


# In[137]:


import pickle
pickle.dump(cv,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




