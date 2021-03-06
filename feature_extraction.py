# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:54:21 2016

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

"""
full_data=pd.read_csv(r"D:/IE Project/sarcasm-data-3000.tsv",delimiter="\t",header=None)

words = set()
with open(r'D:\IE Project\full_words.txt') as f:
    for line in f:
        words.add(line.strip())
solutions = {}
"""
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
    #list_of_extra_hashtags=[]
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
        #list_of_extra_hashtags.append(other_hashtags)
        
        ls.append(temp_ls)    
    return ls,final_emoticon_ls    

#fvs_1_5,emoticons_data=feature_1_to_6(tweets_after_sarcasm2)
#df_1_to_5=pd.DataFrame(fvs_1_5)

#feature7: No of sentences

#emoticon to word mapping i.e replace emoticons by suitable words in the text corpus
#after doing the above, do

#feature8 : sentiment polarity for each sentence
#feature9 : Number of polarity shifts/total number of sentences -1
#feature10 : subjectivity of each sentence
#feature11 : subjectivity shift for all sentences
#feature11 : polarity similarity comparison b/w the tweet and the remaning hashtags 
#feature12 : subjectivity similarity comparison b/w the tweet and the remaining hashtags
"""
def sentiments_features(data):
    for i in data:
        d=TextBlob(i)
"""
#replace multiple occurrences with one ocurence re.sub(r'(\W)(?=\1)', '', line)

#to do find : occurences in time and date and don't do split if they are there

def word_replacement(each_tweet,model_for_simcomp,wiki_words):
    other_hashtags_words=converting_other_hashtags_into_words(each_tweet).split()
    words_in_tweet_wo_hashtag_info=each_tweet.split()
    words_in_tweet=words_in_tweet_wo_hashtag_info+other_hashtags_words
    #words_in_tweet_not_in_wiki=[i for i in words_in_the_tweet if i not in wiki_words]
    ls_of_corrected_words=[]
    for i in words_in_tweet:
       if i in wiki_words:
           ls_of_corrected_words.append(i)
       else:    
           query = i
           list_spellchecks=Word(query).spellcheck()
           most_possible_word=list_spellchecks[0]
           if most_possible_word[1]!=0:
              ls_of_corrected_words.append(most_possible_word[0])
           else:   
              url = 'http://api.urbandictionary.com/v0/define?term=%s' % (query)
              response = urllib.urlopen(url)
              data = json.loads(response.read())
              definition = data['tags']
              if len(definition)!=0:
                 ls_of_corrected_words.append(best_match_for_word(query,definition,model_for_simcomp))
              else:
                 ls_of_corrected_words.append(query)
                 
    return " ".join([i for i in ls_of_corrected_words])          

def best_match_for_word(actual_word,list_of_words_returned_by_urbandictionary_api,model_to_compare_similarity):
    our_word=actual_word
    list_comp=list_of_words_returned_by_urbandictionary_api
    cur_model=model_to_compare_similarity        
    max_sim=0.0
    optimal_word=""
    for i in list_comp:
        if i in cur_model.vocab():
            cur_sim=cur_model.similarity(i,our_word)
            if cur_sim>max_sim:
                max_sim=cur_sim
                optimal_word=i
    return optimal_word            
    
def optimal_split(sentence,values_list):
    #print values_list
    sentence=re.sub(r'(\W)(?=\1)','',sentence)
    #total_len=0
    #print sentence
    max_score=10000
    best_op="nothing"
    #print values_list
    if len(values_list)==0:
        return TextBlob(sentence).sentences
    for i in values_list:
        #cur_excluding_list=[j for j in values_list if j!=i]
        #cur_sentence_wo_excluding_values=" ".join(k for k in sentence.split(i) if k not in cur_excluding_list)
        ls=sentence.split(i)
        for j in ls:
            for k in j:
                if k in values_list:
                    j.replace(k,"")
        #print sentence
        #print ls            
        score=split_criterion(sentence,ls)
        #print score
        if score<max_score:
            max_score=score
            best_op=i
    #print best_op    
    #print [TextBlob(i).sentences[0] for i in dct["opt"]]        
    return sentence.split(best_op)        


#change the split criterion metric coz it's not working for some examples like
#optimal_split("This is a test and we ! are fine",["!","and"])
#0.666666666667
#0.5
#[Sentence("This is a test and we "), Sentence(" are fine")]    

def split_criterion(sen,proposed_split):
    assert isinstance(sen,str),"Looks like a string is not passed as the first argument to the function that scores the split proposed"
    assert isinstance(proposed_split,list),"Looks like a list is not passed as the proposed split to the function split criterion" 
    criterion_score=0
    num_parts=len(proposed_split)
    #total_len=len(sen.split())
    len_proposed_split=0    
    for i in proposed_split:
        len_proposed_split+=len(i.split())
    for i in proposed_split:
        if len(i.split())==0:
           x=1
        else:
           x=len([k for k in i.split() if k!="" or k!=" "])
        criterion_score+=abs(1.0/num_parts-x/float(len_proposed_split))
    return criterion_score     
        
def sentence_split(sentence):
    ops=["!",":",";",".","and"]
    sentence=re.sub(r'(\W)(?=\1)','',sentence)
    #print sentence
    sentence=multiple_hashtag_deletion(sentence,"#")[0]
    #print sentence
    d=TextBlob(sentence)
    res=d.sentences
    if len(res)!=1:
        return res
    else:
        #print "in else"
        for i in ops:
            if i not in sentence:
                ops[ops.index(i)]=""
                #ops.remove(i)
        new_ops=[i for i in ops if i !=""]        
        return optimal_split(sentence,new_ops)   
        
def polarity_subj_and_shifts(tweet,other_hashtags_bool):
    tweet=re.sub(r'(\W)(?=\1)','',tweet)  
    tb_tweet=TextBlob(tweet)
    p=tb_tweet.polarity
    s=tb_tweet.subjectivity
    #print p
    #print s
    if other_hashtags_bool!=1:   
       tweet_parts=sentence_split(tweet)
       #print "here"
       #print tweet_parts
       shifts_ls=[]
       #pol_sub_ls=[]
       for i in tweet_parts:
           temp_ls=[]
           tb_tweet_part=i
           #print tb_tweet_part
           if isinstance(tb_tweet_part,str):
               tb_tweet_part=TextBlob(tb_tweet_part)
           pol=tb_tweet_part.polarity
           sub=tb_tweet_part.subjectivity
           #print pol
           #print sub
           temp_ls.append(pol)
           temp_ls.append(sub)
           shifts_ls.append(temp_ls)
       pd_df=pd.DataFrame(shifts_ls,index=None)
       #print pd_df.columns
       #print pd_df
       shifts_ls=[]
       for i in pd_df.columns:
           cur_col=list(pd_df[i])
           #print cur_col
           temp_sum=0.0
           for j in range(len(cur_col)):
              if j+1<len(cur_col):
                  temp_sum+=ds_metric.euclidean(cur_col[j],cur_col[j+1])
           shifts_ls.append(temp_sum)        
       #sprint p,s,shifts_ls[0],shifts_ls[1]    
       return p,s,shifts_ls[0],shifts_ls[1]
    elif other_hashtags_bool==1:
         return p,s
"""
def find_words(instring):
    # First check if instring is in the dictionnary
    if instring in words:
        #print instring+" found in dict"
        return [instring]
    # No... But maybe it's a result we already computed
    if instring in solutions:
        #print instring+" found in previous solutions"
        return solutions[instring]
    # Nope. Try to split the string at all position to recursively search for results
    best_solution = None
    for i in range(1, len(instring) - 1):
        #print "instring[:i] "+instring[:i]
        #print "instring[i:] "+instring[i:]
        part1 = find_words(instring[:i])
        part2 = find_words(instring[i:])
        # Both parts MUST have a solution
        #print "part1 "+str(part1)
        #print "part2 "+str(part2)
        if part1 is None or part2 is None:
            #print "one of them is none"
            continue
        solution = part1 + part2
        #print "solution " + str(solution)
        # Is the solution found "better" than the previous one?
        if best_solution is None or len(solution) < len(best_solution):
            best_solution = solution
        #print best_solution    
    # Remember (memoize) this solution to avoid having to recompute it
    solutions[instring] = best_solution
    #print best_solution
    return best_solution
"""

def find_words(instring, prefix = '', words = None):
    if not instring:
        return []
    if words is None:
        words = set()
        with open(r'D:\IE Project\full_words.txt') as f:
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

def converting_other_hashtags_into_words(tweet):
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
        if len(ls)==0:
           ls.append(cur_word_str)
        elif len(cur_word_str)>len(ls[0]):
            ls[0]=cur_word_str
    if len(ls)>0:
        return ls[0]
    else:
        return ""
        
def senti_sim_comp_bw_tweet_and_rest_hashtags(tweet,polarity,subj):
    other_hashtags_sentence=converting_other_hashtags_into_words(tweet)    
    #sentence_wo_any_hashtags=multiple_hashtag_deletion(tweet,'#')[0]
    #whole_p,whole_s,pol_shift,sub_shift=polarity_subj_and_shifts(sentence_wo_any_hashtags) 
    #other_hashtags_sentence=" ".join(i for i in other_hashtags[0])
    if  len(other_hashtags_sentence)!=0 and len(other_hashtags_sentence)!=1:
        whole_hashtag_p,whole_hashtag_s=polarity_subj_and_shifts(other_hashtags_sentence,1)
        pol_comp=ds_metric.euclidean(polarity,whole_hashtag_p)
        sub_comp=ds_metric.euclidean(subj,whole_hashtag_s)
        return pol_comp,sub_comp
    else:
        return 0.0,0.0
        
    
def feature_7_to_12(data):
    ls=[]
    c=0
    for i in data:
        c=c+1
        if c%100==0:
            print "in feature 7 to 12 currently data point "+str(c)
        temp_ls=[]
        temp_ls.append(len(sentence_split(i)))
        i_wo_hashtags=multiple_hashtag_deletion(i,"#")[0]
        p,s,ps,ss=polarity_subj_and_shifts(i_wo_hashtags,2)
        sent_sim_score,sub_sim_score=senti_sim_comp_bw_tweet_and_rest_hashtags(i,p,s)
        temp_ls.append(p)
        temp_ls.append(ps)
        temp_ls.append(s)
        temp_ls.append(ss)
        temp_ls.append(sent_sim_score)
        temp_ls.append(sub_sim_score)
        ls.append(temp_ls)
    return ls

def emoticons_feature_one_hot_encoding(emoticons_list):
    ls=[]
    for i in emoticons_list:
        s=""
        for j in i:
            s=s+j+"_"
        ls.append(s[:-1])    
    #print ls
    res_df=pd.get_dummies(ls)    
    return res_df
       
def emoticon_extraction_with_re(tweets_file):
    ls=[]
    for i in tweets_file:
        temp_ls=[]
        x=re.findall(r"(?::|:'|;|=)(?:-)?(?:\)|\(|D|P)|(?:\)|\(|:'|;|=)(?:-)?(?::|;|D|P)",i)
        #y=re.findall(r"(?:\)|\(|:'|;|=)(?:-)?(?::|;|D|P)",i)
        for j in x:
            temp_ls.append(j)
        ls.append(temp_ls)
    return ls        
    
if __name__=="__main__":
    full_data=pd.read_csv(r"E:/IE Project/sarcasm-data-3000.tsv",delimiter="\t",header=None)
    words = set() 
    with open(r'D:\IE Project\full_words.txt') as f:
        for line in f:
            words.add(line.strip())
    solutions = {}
    emoticon_dict={":')":"wink",":)":"happy",":(":"sad",":-)":"happy",":-(":"sad",":-P":"playfulness",":P":"playfullness",":/":"criticism",":-/":"criticism",":D":"laughter",":-D":"laughter",";-)":"cheekiness",";)":"cheekiness"}
    tweets_data=list(full_data[2])
    tweets_after_sarcasm2=data_cleaning(tweets_data)
    fvs_1_6,emoticons_data=feature_1_to_6(tweets_after_sarcasm2)
    df_1_to_6=pd.DataFrame(fvs_1_6)
    fvs_7_12=feature_7_to_12(tweets_after_sarcasm2)
    

     
        
            
        
        
       
   

    

