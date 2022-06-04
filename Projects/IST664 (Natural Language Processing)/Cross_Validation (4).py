#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import nltk
from nltk import precision,recall
from nltk.metrics import *
from nltk.classify import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

# Function that uses NLTK to compute evaluation measures (using gold labels and predicted labels), returns precision and recall for each label
def eval_measures(reflist,testlist,labels):
    ref_sets={}
    test_sets={}
    for lab in labels:
        ref_sets[lab]=set()
        test_sets[lab]=set()
    # get gold labels
    for j,label in enumerate(reflist):
        ref_sets[label].add(j)
    # get predicted labels
    for k,label in enumerate(testlist):
        test_sets[label].add(k)
    # lists to return precision and recall for all labels
    precision_list=[]
    recall_list=[]
    #compute precision and recall for all labels using the NLTK functions
    for lab in labels:
        precision_list.append(nltk.metrics.scores.precision(ref_sets[lab],test_sets[lab]))
        recall_list.append(nltk.metrics.scores.recall(ref_sets[lab],test_sets[lab]))
    return (precision_list,recall_list)

# Function to Compute F-Measure:
def Fscore(precision,recall):
    print(precision)
    print(recall)
    if (precision==0) and (recall==0):
        return 0
    else:
        return (2*(precision*recall))/(precision+recall)

# Function to Print Precision, Recall and F-Measure for each Label:
def print_eval(precision_list,recall_list,labels):
    fscore=[]
    num_folds=0
    num=0
    for (index,lab) in enumerate(labels):
        num+=1
        if precision_list[index] is None:
            precision_list[index]=0
        if recall_list[index] is None:
            recall_list[index]=0
        fscore.append(Fscore(precision_list[index],recall_list[index]))
        if fscore[num_folds]==0:
            num-=1
        num_folds+=1
    print('Average Precision',(sum(precision_list))/num_folds)
    print('Average Recall:',(sum(recall_list))/num_folds)
    print('F-score:',(sum(fscore))/num)

# Naive-Bayes Classifier Function:
def naive_bayes(num_folds,featuresets,labels):
    subset_size=int(len(featuresets)/num_folds)
    reflist=[]
    testlist=[]
    accuracy_list=[]
    print("Naive Bayes Classifier")
    for i in range(num_folds):
        print('Start Fold',i)
        test_this_round=featuresets[i*subset_size:][:subset_size]
        train_this_round=featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        classifier=nltk.NaiveBayesClassifier.train(train_this_round)
        accuracy_this_round=nltk.classify.accuracy(classifier, test_this_round)
        print(i,accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
        for (features,label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    print('Done with Cross-Validation')
    print('Mean Accuracy:',(sum(accuracy_list))/num_folds)
    (precision_list,recall_list)=eval_measures(reflist,testlist,labels)
    print_eval(precision_list,recall_list,labels)
    print(' ')
    
# Logistic Regression Classifier Function:
def logisticregression(num_folds,featuresets,labels):
    subset_size=int(len(featuresets)/num_folds)
    reflist=[]
    testlist=[]
    accuracylist=[]
    print('Logistic Regression Classifier')
    for i in range(num_folds):
        print('Start Fold',i)
        test_this_round=featuresets[i*subset_size:][:subset_size]                             
        train_this_round=featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        classifier=SklearnClassifier(LogisticRegression())
        classifier.train(train_this_round)
        accuracy_this_round=nltk.classify.accuracy(classifier,test_this_round)
        print(i,accuracy_this_round)
        for (f,label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
    print('Done with Cross-Validation')
    print('Mean Accuracy: ',(sum(accuracy_list)/num_folds))
    (precision_list,recall_list)=eval_measures(reflist,testlist,labels)
    print_eval(precision_list,recall_list,labels)
    print(' ')


# In[ ]:




