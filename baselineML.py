# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:23:06 2016

@author: Murali
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline,metrics, grid_search 
from numpy import genfromtxt
#stop=stopwords.words("english")

def read_tweets(cleannot,dataset):
    if cleannot=="clean":
       full_text=[]
       with open("E:/Sarcasm detection/new_cleaned_tweets.txt","r") as f:
           for i in f.readlines():
              full_text.append(i[:-2])
       print len(full_text)       
       return full_text 
    else:
        return dataset[2]
        
        
        
if __name__=="__main__":
   clean_or_not=str(raw_input("Want to load clean tweets or the raw ones say clean for cleaned ones anything else for raw ones:"))
   full_data=pd.read_csv(r"E:/Sarcasm detection/sarcasm-data-3000.tsv",delimiter="\t",header=None)
   #full_data=np.delete(full_data,(0),axis=0)
   tweets=read_tweets(clean_or_not,full_data)
   
   print full_data.shape
   full_data[2]=tweets
   targets=list(full_data[1])
   mod_targets=[1 if i=="SARCASM" else 0 for i in targets]
   full_data[1]=mod_targets
   #features_1_5=pd.read_csv("E:/Sarcasm detection/features1_5.csv",delimiter=",",header=None)
   features_1_5=pd.read_csv("E:/Sarcasm detection/features1_5.csv",delimiter=",",header=None)
   features_1_5=features_1_5.drop(features_1_5.index[[0]])
   #features_1_5=np.delete(features_1_5,(0),axis=0)
   """
   full_text=[]
   with open("D:/IE Project/new_cleaned_tweets.txt","r") as f:
        for i in f.readlines():
            full_text.append(i[:-2])
   """        
   #full_text=full_text[0]
   #full_data[2]=full_text    
   full_text=tweets
   del full_data[0]
   x_train,x_test,y_train,y_test=train_test_split(full_data[2],full_data[1],test_size=0.4,stratify=full_data[1])
   len_train=x_train.shape[0]
   len_test=x_test.shape[0] 
   #full_train=[pd.DataFrame(x_train),pd.DataFrame(x_test)]
   tfidf_vec=TfidfVectorizer(analyzer="word",max_features=None,token_pattern=r'\w{1,}',strip_accents='unicode',lowercase=True,ngram_range=(1,3),min_df=2,use_idf=True,smooth_idf=True,norm="l2",sublinear_tf=True)
   train_tfidf_matrix=tfidf_vec.fit_transform(x_train)
   test_tf_idf_matrix=tfidf_vec.transform(x_test)
   train=pd.DataFrame(train_tfidf_matrix.toarray())
   test=pd.DataFrame(test_tf_idf_matrix.toarray())
   train_features= features_1_5[:len_train]
   test_features=features_1_5[len_train:]
   print train_features.shape
   print test_features.shape
   train=np.append(train,features_1_5[:len_train],axis=1)
   test=np.append(test,features_1_5[len_train:],axis=1)
   print "After concat"
   print train.shape
   print test.shape
   print y_train.shape
   print y_test.shape
   svd = TruncatedSVD(algorithm='randomized', random_state=None, tol=0.0)
   scl = StandardScaler()
   lr_model = lm.LogisticRegression(class_weight="balanced",tol=0.00001) 
   clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
                    	     ('lr', lr_model)])
   param_grid = {'svd__n_components' : [200,250,300,350,400],
                 'svd__n_iter':[3,4,5],
                 'lr__C': [10,11,12,13,14,15,16,17],
                  'lr__penalty':["l1","l2"]}
    
    #f  Scorer 
   f_scorer = metrics.make_scorer(f1_score, greater_is_better = True)
    
    # Initialize Grid Search Model
   model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=f_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=3)
                                     
    # Fit Grid Search Model
   model.fit(train, y_train)
   print("Best score: %0.3f" % model.best_score_)
   print("Best parameters set:")
   best_parameters = model.best_estimator_.get_params()
   for param_name in sorted(param_grid.keys()):
   	  print("\t%s: %r" % (param_name, best_parameters[param_name]))
   best_model = model.best_estimator_
   best_model.fit(train,y_train)
   preds = best_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(target_labels,preds,average="weighted")
   
"""
Not cleaned  unigrams
Best score: 0.554
Best parameters set:
	lr__C: 11
	lr__penalty: 'l2'
	svd__n_components: 200
	svd__n_iter: 4
0.575221238938

cleaned unigrams
Best score: 0.521
Best parameters set:
	lr__C: 13
	lr__penalty: 'l2'
	svd__n_components: 250
	svd__n_iter: 4
0.512301013025

not cleaned unigrams and bigrams

Best score: 0.583
Best parameters set:
	lr__C: 17
	lr__penalty: 'l2'
	svd__n_components: 200
	svd__n_iter: 3
0.576368876081

cleaned unigrams and bigrams

Best score: 0.496
Best parameters set:
	lr__C: 12
	lr__penalty: 'l1'
	svd__n_components: 200
	svd__n_iter: 4
0.477011494253

"""   