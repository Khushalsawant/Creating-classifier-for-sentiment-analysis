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


# importing the multiprocessing module 
import multiprocessing 
import os

# importing the threading module 
import threading 

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from nltk.corpus import reuters
from textblob import TextBlob
from textblob import classifiers

import pickle              # import module first

## Here Value of len(reuters.sents()) = 54716 which is equaly divided by 32
def split_into_parts(number, n_parts):
    a_1 = np.array([0.0,1710.0])
    a = np.around((np.linspace(0, number, n_parts+1)[1:]))
    b = np.append(a,a_1)
    b = np.reshape(a, (-1, 2))
    for i in range(len(b)):
        b[i][0] = b[i][0] +1
        print(b[i][0])
    b = np.sort(b,axis=0)
    return b

def create_raw_data_for_classifier(start_pt,end_pt):
    # printing process id 
    #print("ID of process running : {}".format(os.getpid()))
    
    #df_for_raw = pd.DataFrame(columns=['sentences','polarity'])
    pos = 0
    neg = 0
    polarity_list = []
    sentncs_list = []
    
    #for i in range(len(reuters.sents())):
    for i in range(start_pt,end_pt):    
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
    raw_data = list(zip(sentncs_list,polarity_list))
    #print(reutersDf.tail(10))
    print("raw_data len = ",len(raw_data))
    print("Total pos = ",pos," Total Neg =",neg)
    print(raw_data[0])
    
    raw_data = list(zip(sentncs_list,polarity_list))
    #print(reutersDf.tail(10))
    print("raw_data len = ",len(raw_data))
    print("Total pos = ",pos," Total Neg =",neg)
    print(raw_data[0])
    return raw_data


def get_the_classifier_accuracy(raw_data):
    np.random.shuffle(raw_data)
    training = raw_data[:3500]
    testing = raw_data[-3500:]
    
    classifier = classifiers.NaiveBayesClassifier(training)
    
    ## decision tree classifier
    dt_classifier = classifiers.DecisionTreeClassifier(training)
    NaiveBayesClassifier_accuracy = classifier.accuracy(testing)
    DecisionTreeClassifier_accuracy = dt_classifier.accuracy(testing)
    print ("classifier.accuracy = ",classifier.accuracy(testing))
    print ("dt_classifier.accuracy = ",dt_classifier.accuracy(testing))
    return NaiveBayesClassifier_accuracy,DecisionTreeClassifier_accuracy

def convert(n): 
	return str(datetime.timedelta(seconds = n)) 
	

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))

def main_task():
    print("\n Task has been assigned to thread: {}".format(threading.current_thread().name))
    pool = multiprocessing.Pool(processes=6)
    result_list = pool.starmap(create_raw_data_for_classifier,[(0,7000)])#product([(0,100)],repeat=2))
    print("multiprocessing.cpu_count() = ",multiprocessing.cpu_count())
    print("result_list type = ",type(result_list))
    f = open('raw_data_for_classifier.pkl', 'wb')   # Pickle file is newly created where foo1.py is
    pickle.dump(result_list, f,-1)          # dump data to f
    f.close() 
    pool.close()
    pool.join()    

def another_main_task():
    print("\n Task has been assigned to thread: {}".format(threading.current_thread().name))
    pool = multiprocessing.Pool(processes=6) 
    f = open('raw_data_for_classifier.pkl', 'rb')   # 'r' for reading; can be omitted
    raw_data_for_classifier = pickle.load(f)         # load file content as mydict
    f.close()                       

    #print(raw_data_for_classifier)

    #print(result_list)
    result_accuracy1 = pool.starmap(get_the_classifier_accuracy,[raw_data_for_classifier])#product([(0,100)],repeat=2))
    print("multiprocessing.cpu_count() = ",multiprocessing.cpu_count())
    print("result_accuracy1 = ",result_accuracy1)
    pool.close()
    pool.join()    

if __name__ == "__main__": 
    manager = multiprocessing.Manager()
    print("ID of main process: {}".format(os.getpid()))
    print("Main thread name: {}".format(threading.main_thread().name)) 
    
    t1 = threading.Thread(target=main_task,name='create_raw_data_for_classifier')
    t2 = threading.Thread(target=another_main_task,name='raw_data_for_classifier')
    
    t1.start() 
    t1.join()
    
    t2.start() 
    t2.join()
    # both threads completely executed 
    print("both threads completely executed ... Done!") 
    '''
    return_raw_data = manager.list()
    return_dt_classifier_accuracy = manager.list()
    return_NaiveBayesClassifier = manager.list()
    # creating processes 
    p1 = multiprocessing.Process(target=create_raw_data_for_classifier,args=(0,100,return_raw_data))    
    # starting process 1 
    p1.start() 
    # wait until process 1 is finished 
    
    # process IDs 
    print("ID of process p1: {}".format(p1.pid))
    
    p1.join() 
    # both processes finished 
    print("process has finished execution!!") 
    
    # check if processes are alive 
    print("Process p1 is alive: {}".format(p1.is_alive()))
    
    print(p1)
    
    print("Len of return_raw_data",len(return_raw_data))
    
    p2 = multiprocessing.Process(target=get_the_classifier_accuracy,args=(return_raw_data,return_dt_classifier_accuracy,return_NaiveBayesClassifier))    
    
    p2.start() 
    
    # process IDs 
    print("ID of process p2: {}".format(p2.pid))
    
    p2.join()
    
    print("return_dt_classifier_accuracy = ",return_dt_classifier_accuracy)
    
    print("return_NaiveBayesClassifier = ",return_NaiveBayesClassifier)
    
    # check if processes are alive 
    print("Process p2 is alive: {}".format(p2.is_alive()))
    '''
    #pool = multiprocessing.Pool(processes=6)
    #print("multiprocessing.cpu_count() = ",multiprocessing.cpu_count())
    #split_into_parts = pool.starmap(split_into_parts,[(54716,32)])
    #print("Type split_into_parts = ",type(split_into_parts))
    '''
    result_list = pool.starmap(create_raw_data_for_classifier,[(0,10000)])#product([(0,100)],repeat=2))
    print("multiprocessing.cpu_count() = ",multiprocessing.cpu_count())
    print("result_list type = ",type(result_list))
    f = open('raw_data_for_classifier.pkl', 'wb')   # Pickle file is newly created where foo1.py is
    pickle.dump(result_list, f,-1)          # dump data to f
    f.close()    
      
    f = open('raw_data_for_classifier.pkl', 'rb')   # 'r' for reading; can be omitted
    raw_data_for_classifier = pickle.load(f)         # load file content as mydict
    f.close()                       

    #print(raw_data_for_classifier)

    #print(result_list)
    result_accuracy1 = pool.starmap(get_the_classifier_accuracy,[raw_data_for_classifier])#product([(0,100)],repeat=2))
    print("multiprocessing.cpu_count() = ",multiprocessing.cpu_count())
    print("result_accuracy1 = ",result_accuracy1)
    #pool.close()
    #pool.join()
    '''
    n =  time.time() - start_time
    
    print("---Execution Time ---",convert_sec(n))