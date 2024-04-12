#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pickle
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker


# In[14]:


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


# In[15]:


cv = pickle.load(open('/Users/owner/Downloads/kiwi/vectorizer.pkl','rb'))
model = pickle.load(open('/Users/owner/Downloads/kiwi/model.pkl','rb'))


# In[16]:


st.title("Cyber Bullying Type Classifier")
input_text = st.text_input("Enter the message")


# In[18]:


if st.button('predict'):
    #1. Preprocess
    processed_text = transform_text(input_text)
    
    #2. vectorize
    vector_input = cv.transform([processed_text])
    
    #3. Predict
    result = model.predict(vector_input)[0]
    
    #4. Display
    if result == 1:
        st.header("Sexual Harassment")
    elif result == 2:
        st.header("Cyberstalking")
    elif result == 3:
        st.header("Doxing")
    elif result == 4:
        st.header("Revenge Porn")
    elif result == 5:
        st.header("Slut Shaming")


# In[ ]:




