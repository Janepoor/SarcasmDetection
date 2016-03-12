# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:45:09 2016

@author: Murali
"""

import pandas as pd
#import gensim as gs
import os
import shutil
import hashlib
from sys import platform
import gensim

#add_data1_regular=pd.read_csv(r"D:/IE Project/additional_sarcasm_data/twitDB_regular.csv",delimiter=",",header= None)
#add_data1_sarcasm=pd.read_csv(r"D:/IE Project/additional_sarcasm_data/twitDB_sarcasm.csv",delimiter=",",header= None)
add_data1_regular=[]
add_data1_sarcasm=[]
add_data2=[]
with open(r"D:/IE Project/additional_sarcasm_data/twitDB_regular.txt","r") as f:
    for j in f.readlines():
        add_data1_regular.append(j[:-15])
        
with open(r"D:/IE Project/additional_sarcasm_data/twitDB_sarcasm.txt","r") as f:
    for j in f.readlines():
        add_data1_sarcasm.append(j[:-12])
        
with open(r"D:/IE Project/additional_sarcasm_data/pfs_tweets.txt","r") as f:
    for j in f.readlines():
        add_data2.append(j[:-1])

#Load the saved glove twitter model.
#glove_model=gensim.models.word2vec.Word2Vec.load_word2vec_format(gensim_file,binary=False) #GloVe Model
print "Loading Glove model, this can take some time"
glove_twitter=gensim.models.word2vec.Word2Vec.load_word2vec_format("D:\IE Project\glove_twitter_model.txt",binary=False)

"""
print model.most_similar(positive=['australia'], topn=10)
print model.similarity('woman', 'man') 
"""