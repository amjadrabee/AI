import numpy as np
import pandas as pd
import tensorflow as tf
from string import punctuation
from tensorflow.keras import layers, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')

df = pd.read_csv('/Users/amjadrabee/PycharmProjects/ObjectDetection/school_projects/Language Detection.csv')
df.head()

def preprocess_text(text):
    """
    Preprocess text data 
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = ''.join(char for char in text if char not in punctuation)
    
 
    text = ''.join(char for char in text if not char.isdigit())
    
    words = word_tokenize(text)
    
    
    stemmer = PorterStemmer()
    
    stemmed_words = [stemmer.stem(word) for word in words]
    

    preprocessed_text = ' '.join(stemmed_words)
    
    return preprocessed_text

df['preprocessed_text'] = df['Text'].apply(preprocess_text)
df['preprocessed_text'].head()

text = df['preprocessed_text']
language = df['Language']

vectorizer = CountVectorizer()
label_encoder = LabelEncoder()

text_v = vectorizer.fit_transform(text)
language_v = label_encoder.fit_transform(language)

model = MultinomialNB()
model.fit(text_v,language_v)

def language_detector(input_xlanguage):
    
    input_xlanguage_v = vectorizer.transform([input_xlanguage])

    language_detected = (model.predict(input_xlanguage_v)[0])

    return language_detected

def predict(input_text):
    
    predicted_language = label_encoder.inverse_transform((language_detector(input_text)).reshape(-1))
    print(f"The predicted language is: {predicted_language}")

predict("Привет, это тест")
predict("we are boys")
predict("انا مصري ")
predict("mon petit.")