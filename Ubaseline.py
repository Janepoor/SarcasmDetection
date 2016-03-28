# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:03:53 2016

@author: Murali
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:00:34 2016

@author: Murali
"""

import pandas as pd
import numpy as np
import nltk
import re
import urllib
import json
import scipy.spatial.distance as ds_metric
from textblob import TextBlob,Word
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim as gs
import gensim
from fuzzywuzzy import fuzz
#import glovefile_to_w2v as gw

#feature1 : question_mark presence
#feature2 : Presence of hastags other than #sarcasm tag
#feature3 : presence of Http
#feature4 : exclamation mark presence
#featurex : extract other hashtags
#feature5 : number of capitalized words in the tweet

#presence of emoticons 
#for finding winking emoticons "match" if re.match('^(:\'\(|:\'\))+$',":')") else "no"
#for finding smileys and other emoticons "match" if re.match('^(:\(|:\))+$',":)") else "no"

#emoticon_dict={":')":"wink",":)":"happy",":(":"sad",":-)":"happy",":-(":"sad",":-P":"playfulness",":P":"playfullness",":/":"criticism",":-/":"criticism",":D":"laughter",":-D":"laughter",";-)":"cheekiness",";)":"cheekiness"}
#tweets_data=list(full_data[2])

def multiple_hashtag_deletion(sentence,hashtag):
    """
    Input: A tweet(sentence) and a symbol(hashtag) which can be # or anything like #xyz and 
    Output:returns the tweet as string after deleting all the occurrences of the hashtag symbol and
           also the list of hashtags that match the hashtag
    """
    sentence=sentence+" "
    #rexp=re.compile('[^a-zA-Z]')
    #sentence=rexp.sub(' ',sentence)
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
    lss=sentence.split()
    for i in lss:
        if i.startswith(("@")):
            lss.remove(i)
    sentence=" ".join([i for i in lss])        
    return sentence,ls_of_hashtags    
        

def data_cleaning(data):
    """
    Input  :  A list or iterator containg raw tweets and for each element in the list calls the above function and 
    Output : returns a list containg all the tweets after the deletion of hashtags from each of them
    Function calls : calls multiple_hashtag_deletion for each tweet in the data for cleaning    
    """
    ls=[]
    for i in data:
        ls.append(multiple_hashtag_deletion(i,"#sarcasm")[0])
    return ls     
    
#tweets_after_sarcasm2=data_cleaning(tweets_data)            

def feature_1_to_6(data):
    """
    Input : Cleaned tweets after deleting sarcasm hashtag (output of the data_cleaning function with #sarcasm as hashtag parameter)
    Ouput : List of lists where each element is a list of length 5 corresponding to the features told above
            and 
            List of lists where each element is a list of emoticons present in each tweet in the datalist
    """
    ls=[]
    #list_of_extra_hashtags=[]
    final_emoticon_ls=[]
    for i in data:
        #print i
        temp_ls=[]
        emoticon_ls=[]
        temp_ls.append(float(i.count("?"))/len(i.split()) if len(i.split())!=0 else 0.0)
        temp_ls.append(1.0 if "http" in i or "Http" in i else 0.0)
        temp_ls.append(float(i.count("!"))/len(i.split()) if len(i.split())!=0 else 0.0)
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
        #list_of_extra_hashtags.append(other_hashtags)
        
        ls.append(temp_ls)    
    return ls,final_emoticon_ls    
    
#tweet_slwords_replacement= lambda sl_words,repl_words,tweet_ls,ls_of_wikiwords: [repl_words[sl_words.index(tweet_ls[i])] if tweet_ls[i] in sl_words else tweet_ls[i] for i in range(len(tweet_ls))]  

def tweet_slwords_replacement(sl_words,repl_words,tweet_ls,ls_of_wikiwords):
    ls=[]
    for i in range(len(tweet_ls)):
        if i not in ls_of_wikiwords:
            if tweet_ls[i] in sl_words:
               ls.append(repl_words[sl_words.index(tweet_ls[i])])
            else:
                 ls.append(tweet_ls[i])
        else:
            ls.append(tweet_ls[i])
    return ls             
                
#repl_words[sl_words.index(tweet_ls[i])] if tweet_ls[i] in sl_words else tweet_ls[i]
        
def word_replacement(each_tweet,model_for_simcomp,wiki_words,slangwords_df):
    """
    Inputs: single tweet after #sarcasm deletion, 
            word2vec model for comparing similarity,
            list of wiki words,
            dataset of slangwords downloaded from github
    Output: A string representation of the input tweet after correcting words in it.        
    Function calls: converting_other_hashtags_into_words in order to get the words present in hashtags other than sarcasm
                   best_match_for_word for words not found in wiki words list
    """
    rexp=re.compile('[^a-zA-Z]')
    slang_words_ls=list(slangwords_df[0])
    replacement_words_ls=list(slangwords_df[1])
    other_hashtags_words=converting_other_hashtags_into_words(each_tweet,"all").split()
    words_in_tweet_wo_hashtag_info=multiple_hashtag_deletion(each_tweet,"#")[0].split()
    words_in_tweet=words_in_tweet_wo_hashtag_info+other_hashtags_words
    
    words_in_tweet2=[rexp.sub(" ",i).split()[0].lower() if len(rexp.sub(" ",i).split())!=0 else " " for i in words_in_tweet]
    words_in_tweet=words_in_tweet2
    #[ if i in wiki_words for i in words_tweet
    words_in_tweet=tweet_slwords_replacement(slang_words_ls,replacement_words_ls,words_in_tweet,wiki_words)
    #words_in_tweet_not_in_wiki=[i for i in words_in_the_tweet if i not in wiki_words]
    ls_of_corrected_words=[]
    #print each_tweet
    for i in words_in_tweet:
        il=i.lower()
        if il in wiki_words or il in model_for_simcomp.vocab.keys() :
           ls_of_corrected_words.append(i)
        else:    
           query = i
           list_spellchecks=Word(query).spellcheck()
           most_possible_word=list_spellchecks[0]
           if most_possible_word[1]>0.5:
              ls_of_corrected_words.append(most_possible_word[0])
           else:   
              url = 'http://api.urbandictionary.com/v0/define?term=%s' % (query)
              response = urllib.urlopen(url)
              data = json.loads(response.read())
              #print data
              if 'tags' not in data.keys():
                  definition=[]
              else:    
                  definition = data['tags']
              if len(definition)!=0:
                 ls_of_corrected_words.append(best_match_for_word(query,definition,model_for_simcomp))
              else:
                 ls_of_corrected_words.append(query)
    #print " ".join([i for i in ls_of_corrected_words])              
    return " ".join([i for i in ls_of_corrected_words])          

def best_match_for_word(actual_word,list_of_words_returned_by_urbandictionary_api,model_to_compare_similarity):
    """
    Input: Word to find the match,
           List of words returned by urban dict api
           Word2Vec model to compare similarity
    Output: Out of all the words present in the list of words returned by urban dict api, return the best match
    Function calls: fuzz.ratio to find out edit distance       
    
    """    
    our_word=actual_word
    list_comp=list_of_words_returned_by_urbandictionary_api
    cur_model=model_to_compare_similarity        
    max_sim=0.0
    optimal_word=""
    for i in list_comp:
        if i in cur_model.vocab.keys() and our_word in cur_model.vocab.keys():
            cur_sim=cur_model.similarity(i,our_word)
        else:
            cur_sim=0.0
        fuzz_sim=float(fuzz.ratio(i,our_word))/100
        tot_sim=cur_sim+fuzz_sim
        if tot_sim>max_sim:
            max_sim=tot_sim
            optimal_word=i
    return optimal_word            

def find_words(instring, prefix = '', words = None):
    if not instring:
        return []
    if words is None:
        words = set()
        with open(r'E:\Sarcasm detection\full_words.txt') as f:
            for line in f:
                words.add(line.strip())
    if (not prefix) and (instring in words):
        return [instring]
    prefix, suffix = prefix + instring[0], instring[1:]
    solutions = []
    # Case 1: prefix in solution
    if prefix in words:
        try:
            #print "----------prefix in words---------------"
            #print prefix,suffix
            solutions.append([prefix] + find_words(suffix, '', words))
        except ValueError:
            pass
    # Case 2: prefix not in solution
    try:
        #print "-------------prefix not in words------------------"
        #print prefix,suffix
        solutions.append(find_words(suffix, prefix, words))
    except ValueError:
        pass
    if solutions:
        #print "---------------solutions---------------"
        #print solutions
        return sorted(solutions,
                      key = lambda solution: [len(word) for word in solution],
                      reverse = True)[0]
    else:
        raise ValueError('no solution')

def converting_other_hashtags_into_words(tweet,best_or_all):
    list_of_all_hashtags=multiple_hashtag_deletion(tweet,"#")[1]
    list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower=[i[1:].lower() for i in list_of_all_hashtags if i[1:].lower()!="sarcasm"]
    ls=[]
    #print list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower
    for i in range(len(list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower)):
        if list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower[i]!="" and list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower[i][-1]==".":
            list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower[i]=list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower[i][:-1]  
    for i in list_of_hashtags_wo_hashsym_sarcasm_and_are_in_lower:
        solutions = {}
        list_of_words=find_words(i)
        #print tweet
        tb_spellcheck=Word(i).spellcheck()
        if tb_spellcheck[0][1]==1.0:
            list_of_words=[]
            list_of_words.append(tb_spellcheck[0][0])
        #print list_of_words
        cur_word_str=" ".join(j for j in list_of_words if len(j)!=1)
        #print cur_word_str
        #there might be more than 1 hashtags and hence we store only the hashtag that is the longest since it has the most information
        if best_or_all=="best":
           if len(ls)==0:
              ls.append(cur_word_str)
           elif len(cur_word_str)>len(ls[0]):
               ls[0]=cur_word_str
        else:
            ls.append(cur_word_str)
    if len(ls)>0:
        if best_or_all=="best":
           return ls[0]
        else:
            return " ".join(j for j in ls if j!="" or j!=" ")
    else:
        return ""
        
        
if __name__=="__main__":
    full_data=pd.read_csv(r"E:/Sarcasm detection/sarcasm-data-3000.tsv",delimiter="\t",header=None)
    twitter_slangwords_data=pd.read_csv(r"E:/Sarcasm detection/twitternoslangwords.csv",delimiter="\t",header=None)       
    twitter_slangwords_data.ix[1881][1]="http"
    twitter_slangwords_data.ix[2178][1]="i am"
    words = set() 
    with open(r'E:\Sarcasm detection\full_words.txt') as f:
        for line in f:
            words.add(line.strip())
    solutions = {}
    emoticon_dict={":')":"wink",":)":"happy",":(":"sad",":-)":"happy",":-(":"sad",":-P":"playfulness",":P":"playfullness",":/":"criticism",":-/":"criticism",":D":"laughter",":-D":"laughter",";-)":"cheekiness",";)":"cheekiness"}
    tweets_data=list(full_data[2])
    tweets_after_sarcasm2=data_cleaning(tweets_data)
    fvs_1_6,emoticons_data=feature_1_to_6(tweets_after_sarcasm2)
    df_1_to_6=pd.DataFrame(fvs_1_6)
    word_repl_w2vmodel=gensim.models.word2vec.Word2Vec.load_word2vec_format("E:\Sarcasm detection\glove_twitter_model.txt",binary=False)
    tweets_after_word_replacement=[word_replacement(j,word_repl_w2vmodel,words,twitter_slangwords_data) for j in tweets_after_sarcasm2] 
    #fvs_7_12=feature_7_to_12(tweets_after_sarcasm2)        