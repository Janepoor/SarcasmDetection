# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:54:21 2016

@author: Murali
"""

import pandas as pd
import numpy as np
import nltk
import re
import scipy.spatial.distance as ds_metric
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim as gs
import breaking_sentences_without_spaces_into_words_2 as bsw

full_data=pd.read_csv(r"D:/IE Project/sarcasm-data-3000.tsv",delimiter="\t",header=None)

words = set()
with open(r'D:\IE Project\full_words.txt') as f:
    for line in f:
        words.add(line.strip())
#solutions = {}
#feature1 : question_mark presence
#feature2 : Presence of hastags other than #sarcasm tag
#feature3 : presence of Http
#feature4 : exclamation mark presence
#feature5 : If other hashtags exist binary 
#feature5 : extract other hashtags
#feature6 : presence of a capitalized word in the tweet

#presence of emoticons 
#for finding winking emoticons "match" if re.match('^(:\'\(|:\'\))+$',":')") else "no"
#for finding smileys and other emoticons "match" if re.match('^(:\(|:\))+$',":)") else "no"

emoticon_dict={":')":"wink",":)":"happy",":(":"sad",":-)":"happy",":-(":"sad",":-P":"playfulness",":P":"playfullness",":/":"criticism",":-/":"criticism",":D":"laughter",":-D":"laughter",";-)":"cheekiness",";)":"cheekiness"}
tweets_data=list(full_data[2])

def multiple_hashtag_deletion(sentence,hashtag):
    sentence=sentence+" "
    lower_sent=sentence.lower()
    c=lower_sent.count(hashtag)
    ls_of_hashtags=[]
    while c>0:
        ind=lower_sent.find(hashtag)
        sp=lower_sent.find(" ",ind+1)
        ls_of_hashtags.append(sentence[ind:sp])
        sentence=sentence[:ind]+sentence[sp:]
        lower_sent=lower_sent[:ind]+lower_sent[sp:]
        c=c-1
    return sentence,ls_of_hashtags    
        

def data_cleaning(data):
    ls=[]
    for i in data:
        ls.append(multiple_hashtag_deletion(i,"#sarcasm")[0])
    return ls     
    
tweets_after_sarcasm2=data_cleaning(tweets_data)            

def feature_1_to_6(data):
    ls=[]
    list_of_extra_hashtags=[]
    final_emoticon_ls=[]
    for i in data:
        temp_ls=[]
        emoticon_ls=[]
        temp_ls.append(float(i.count("?"))/len(i.split()))
        temp_ls.append(1.0 if "http" in i or "Http" in i else 0.0)
        temp_ls.append(float(i.count("!"))/len(i.split()))
        #print i
        other_hashtags=[j[1:] for j in i.split() if j.startswith("#")]
        temp_ls.append(1.0 if len(other_hashtags)!=0 else 0.0)
        temp_ls.append(sum([1 if j.isupper() else 0 for j in i.split()]))
        for k in emoticon_dict:
            if k in i:
                emoticon_ls.append(emoticon_dict[k])
        if len(emoticon_ls)==0:
            emoticon_ls.append("No")
        final_emoticon_ls.append(emoticon_ls)
        list_of_extra_hashtags.append(other_hashtags)
        
        ls.append(temp_ls)    
    return ls,list_of_extra_hashtags,final_emoticon_ls    

fvs_1_5,extra_hashtags_data,emoticons_data=feature_1_to_6(tweets_after_sarcasm2)
df_1_to_5=pd.DataFrame(fvs_1_5)

#feature7: No of sentences

#emoticon to word mapping i.e replace emoticons by suitable words in the text corpus
#after doing the above, do

#feature8 : sentiment polarity for each sentence
#feature9 : Number of polarity shifts/total number of sentences -1
#feature10 : subjectivity of each sentence
#feature11 : subjectivity shift for all sentences
#feature11 : sentiment similarity comparison b/w the tweet and the remaning hashtags 
"""
def sentiments_features(data):
    for i in data:
        d=TextBlob(i)
"""
#replace multiple occurrences with one ocurence re.sub(r'(\W)(?=\1)', '', line)
def optimal_split(sentence,values_list):
    sentence=re.sub(r'(\W)(?=\1)','',sentence)
    total_len=len(sentence.split())
    dct={}
    for i in values_list:
        ls=sentence.split(i)
        if len(ls)<total_len:
            total_len=len(ls)
            dct["opt"]=ls
            
    return dct["opt"]        
        
        

def sentence_split(sentence):
    ops=["!",":",";","."]
    sentence=re.sub(r'(\W)(?=\1)','',sentence)
    sentence=multiple_hashtag_deletion(sentence,"#")[0]
    d=TextBlob(sentence)
    res=d.sentences
    if len(res)!=1:
        return res
    else:
        for i in ops:
            if i not in sentence:
                ops.remove(i)
        return optimal_split(sentence,ops)   
        
def polarity_subj_and_shifts(tweet):
    tweet=re.sub(r'(\W)(?=\1)','',tweet)
    tweet_parts=sentence_split(tweet)
    tb_tweet=TextBlob(tweet)
    shifts_ls=[]
    #pol_sub_ls=[]
    p=tb_tweet.polarity()
    s=tb_tweet.subjectivity()
    for i in tweet_parts:
        temp_ls=[]
        tb_tweet_part=TextBlob(i)
        pol=tb_tweet_part.polarity()
        sub=tb_tweet_part.subjectivity()
        temp_ls.append(pol)
        temp_ls.append(sub)
        shifts_ls.append(temp_ls)
    pd_df=pd.DataFrame(shifts_ls,index=None) 
    for i in pd_df.columns:
        cur_col=list(pd_df[i])
        temp_sum=0.0
        for j in range(len(cur_col)):
            if j+1<len(cur_col):
                temp_sum+=ds_metric.euclidean(cur_col[j],cur_col[j+1])
        shifts_ls.append(temp_sum)        
    return p,s,shifts_ls[0],shifts_ls[1]


    
def converting_other_hashtags_into_words(tweet):
    list_of_all_hashtags=multiple_hashtag_deletion(tweet,"#")[1]
    list_of_all_hashtags_wo_hash_symbol=[i[1:]for i in list_of_all_hashtags]
    list_of_hashtags_lower=[i.lower() for i in list_of_all_hashtags_wo_hash_symbol]
    list_of_hashtags_wo_sarcasm=[i for i in list_of_hashtags_lower if i!="sarcasm"]
    ls=[]
    for i in list_of_hashtags_wo_sarcasm:
        solutions = {}
        list_of_words=bsw(i)
        cur_word_str=" ".join(j for j in list_of_words)
        ls.append(cur_word_str)
    return ls
        
def senti_sim_comp_bw_tweet_and_rest_hashtags(tweet):
    other_hashtags=converting_other_hashtags_into_words(tweet)    
    sentence_wo_any_hashtags=multiple_hashtag_deletion(tweet,'#')[0]
        

def feature_7_to_11(data):
    ls=[]
    for i in data:
        temp_ls=[]
        temp_ls.append(len(sentence_split(i)))
        p,ps,s,ss=polarity_subj_and_shifts(i)
        sent_sim_score=senti_sim_comp_bw_tweet_and_rest_hashtags(i)
        temp_ls.append(p)
        temp_ls.append(ps)
        temp_ls.append(s)
        temp_ls.append(ss)
        ls.append(temp_ls)
    return ls
     
        
            
        
        
       
   

    

