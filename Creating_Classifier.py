# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:29:28 2019

@author: khushal
"""


'''
Idea for generating own classifier take from below articles,

https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/

https://www.cs.bgu.ac.il/~elhadad/nlp16/ReutersDataset.html

'''
import time
# into hours, minutes and seconds 
import datetime 

start_time = time.time()

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


from nltk.corpus import reuters

#print("reuters.words = ",reuters.words())
#print("reuters.categories = ",reuters.categories())
#print("reuters.sents = ",reuters.sents())

'''
hotel_rev = ["Great place to be when you are in Bangalore.",
"The place was being renovated when I visited so the seating was limited.",
"Loved the ambience, loved the food"]

sid = SentimentIntensityAnalyzer()
for sentence in hotel_rev:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for j in ss:
        print('{0}: {1}, '.format(j, ss[j]), end='\n ')
'''

# Raw corpus of Reuters Data

# Extract fileids from the reuters corpus
fileids = reuters.fileids()

# Initialize empty lists to store categories and raw text
categories = []
text = []

# Loop through each file id and collect each files categories and raw text
for file in fileids:
    categories.append(reuters.categories(file))
    text.append(reuters.raw(file))

# Combine lists into pandas dataframe. reutersDf is the final dataframe.
reutersDf = pd.DataFrame({'ids':fileids, 'categories':categories, 'text':text})

#print(reutersDf['text'])
reutersDf['polarity'] = 'pos'

from textblob import TextBlob
from textblob import classifiers

df_for_raw = pd.DataFrame(columns=['sentences','polarity'])
pos = 0
neg = 0
polarity_list = []
sentncs_list = []
for i in range(len(reutersDf)):
    #print(type(reutersDf['text'][i]))
    #print(reutersDf['text'][i])
    #blob = TextBlob(reutersDf['text'][i])
    text_token = sent_tokenize(reutersDf['text'][i])
    for j in range(len(text_token)):
        sentncs = text_token[j]
        sentncs_list.append(sentncs)
        blob = TextBlob(sentncs)
        if blob.sentiment.polarity > 0:
            #print('Positive: ', round(blob.sentiment.polarity,2))
            #reutersDf['polarity'].loc[ reutersDf.text == reutersDf['text'][i] ] = 'pos'
            polarity_list.append('pos')
            pos = pos + 1
        elif blob.sentiment.polarity < 0:
            #print('Negative: ', round(blob.sentiment.polarity,2))
            #reutersDf['polarity'].loc[ reutersDf.text == reutersDf['text'][i] ] = 'neg'
            polarity_list.append('neg')
            neg = neg + 1

raw_data = list(zip(sentncs_list,polarity_list))
print(reutersDf.tail(10))
print("raw_data len = ",len(raw_data))
print("Total pos = ",pos," Total Neg =",neg)
print(raw_data[0])
#text_list = reutersDf['text'].values.tolist()
#polarity_list = reutersDf['polarity'].values.tolist()

#raw_data_for_classifier = list(zip(text_list,polarity_list))
#print("raw_data_for_classifier = ",raw_data_for_classifier)

training = raw_data[18831]
testing = raw_data[-11790]

classifier = classifiers.NaiveBayesClassifier(training)

## decision tree classifier
dt_classifier = classifiers.DecisionTreeClassifier(training)

print ("classifier.accuracy = ",classifier.accuracy(testing))
print ("dt_classifier.accuracy = ",dt_classifier.accuracy(testing))

def convert(n): 
	return str(datetime.timedelta(seconds = n)) 
	
n =  time.time() - start_time

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))

print("---Execution Time ---",convert_sec(n))