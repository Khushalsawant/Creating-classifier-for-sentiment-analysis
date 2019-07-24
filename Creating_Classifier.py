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
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from nltk.corpus import reuters
from textblob import TextBlob
from textblob import classifiers


## Here Value of len(reuters.sents()) = 54716 which is equaly divided by 32
def split_into_parts(number, n_parts):
    a_1 = np.array([0.0,1710.0])
    a = np.around((np.linspace(0, number, n_parts+1)[1:]))
    b = np.reshape(a, (-1, 2))
    #b = np.concatenate(b,a_1)
    print(b)
    for i in range(len(b)):
        b[i][0] = b[i][0] +1
        print(b[i][0])
    b = np.sort(b,axis=0)
    return b

broken_number_of_parts = split_into_parts(54716,32)
print(broken_number_of_parts[0])
#broken_number_of_parts_2D = np.reshape(broken_number_of_parts, (-1, 2))
 


df_for_raw = pd.DataFrame(columns=['sentences','polarity'])
pos = 0
neg = 0
polarity_list = []
sentncs_list = []

#for i in range(len(reuters.sents())):
for i in range(10000):    
    sentncs = " ".join(reuters.sents()[i])
    #print("sentncs = ", sentncs)
    blob = TextBlob(sentncs)
    sentncs_list.append(sentncs)
    if blob.sentiment.polarity > 0:
        polarity_list.append('pos')
        pos = pos + 1
    elif blob.sentiment.polarity < 0:
        polarity_list.append('neg')
        neg = neg + 1
'''
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
'''
raw_data = list(zip(sentncs_list,polarity_list))
#print(reutersDf.tail(10))
print("raw_data len = ",len(raw_data))
print("Total pos = ",pos," Total Neg =",neg)
print(raw_data[0])
#text_list = reutersDf['text'].values.tolist()
#polarity_list = reutersDf['polarity'].values.tolist()

#raw_data_for_classifier = list(zip(text_list,polarity_list))
#print("raw_data_for_classifier = ",raw_data_for_classifier)

np.random.shuffle(raw_data)
training = raw_data[:5000]
testing = raw_data[-5000:]

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