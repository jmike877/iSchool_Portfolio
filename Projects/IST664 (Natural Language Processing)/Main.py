#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  f
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
import os
import sys
import random
import nltk
import re
from nltk.corpus import stopwords 
from nltk import FreqDist
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

#Import Cross_Validation:
import Cross_Validation

#Import saveFeaturesets:
import saveFeaturesets

#Import sentiment_read_LIWC_pos_neg_words:
import sentiment_read_LIWC_pos_neg_words

#Import sentiment_read_subjectivity:
import sentiment_read_subjectivity

# define a feature definition function here
def bag_of_words(wordlist):
    wordlist=nltk.FreqDist(wordlist)
    word_features=[w for (w,c) in wordlist.most_common(200)] 
    return word_features    

def unigram_features(document,word_features):
    d_words=set(document)
    features={}
    for w in word_features:
        features['contains(%s)'%w] = (w in d_words)
    return features
  
def bag_of_words_bigram(wordlist,bigramcount):
    bigram_measures=nltk.collocations.BigramAssocMeasures()
    finder=BigramCollocationFinder.from_words(wordlist,window_size=3)
    finder.apply_freq_filter(3)
    bigramword_features=finder.nbest(bigram_measures.chi_sq, 3000)
    return bigramword_features[:bigramcount]
  
def bigram_features(document,word_features,bigramword_features):
    d_words=set(document)
    d_bigrams=nltk.bigrams(document)
    features={}
    for w in word_features:
        features['contains(%s)'%w]=(w in d_words)
    for b in bigramword_features:
        features['bigram(%s %s)'%b]=(b in d_bigrams)
    return features
  
def negative_features(document,word_features,negationwords):
    features={}
    for w in word_features:
        features['contains({})'.format(w)]=False
        features['contains(NOT{})'.format(w)]=False
    for i in range(0,len(document)):
        word=document[i]
        if ((i+1)<len(document)) and (word in negationwords):
            i+=1
            features['contains(NOT{})'.format(document[i])]=(document[i] in word_features)
        else:
            if ((i+3)<len(document)) and (word.endswith('n') and document[i+1]=="'" and document[i+2]=='t'):
                i+=3
                features['contains(NOT{})'.format(document[i])]=(document[i] in word_features)
            else:
                features['contains({})'.format(word)]=(word in word_features)
    return features

def POS_features(document,word_features):
    d_words=set(document)
    tagged_words=nltk.pos_tag(document)
    features={}
    for w in word_features:
        features['contains({})'.format(w)]=(w in d_words)
    noun_count=0
    verb_count=0
    adjective_count=0
    adverb_count=0
    for (w,t) in tagged_words:
        if t.startswith('N'): 
            noun_count+=1
        if t.startswith('V'): 
            verb_count+=1
        if t.startswith('J'): 
            adjective_count+=1
        if t.startswith('R'): 
            adverb_count+=1
    features['nouns']=noun_count
    features['verbs']=verb_count
    features['adjectives']=adjective_count
    features['adverbs']=adverb_count
    return features

def POS2_features(document,word_features):
    d_words=set(document)
    tagged_words=nltk.pos_tag(document)
    nwords=clean_text(d_words)
    nwords=remove_punctuation(nwords)
    nwords=remove_stopwords(nwords)
    nwords=lemmatizer(nwords)
    nwords=stemmer(nwords)
    features={}
    for w in word_features:
        features['contains({})'.format(w)]=(w in nwords)
    noun_count=0
    verb_count=0
    adjective_count=0
    adverb_count=0
    for (w,t) in tagged_words:
        if t.startswith('N'):
            noun_count+=1
        if t.startswith('V'):
            verb_count+=1
        if t.startswith('J'):
            adjective_count+=1
        if t.startswith('R'):
            adverb_count+=1
    features['nouns']=noun_count
    features['verbs']=verb_count
    features['adjectives']=adjective_count
    features['adverbs']=adverb_count
    return features

def SL_features(document,word_features,SL):
    d_words=set(document)
    features={}
    for w in word_features:
        features['contains({})'.format(w)]=(w in d_words)
    weakPos=0
    strongPos=0
    weakNeg=0
    strongNeg=0
    for w in d_words:
        if w in SL:
            strength,posTag,isStemmed,polarity=SL[w]
            if strength=='weaksubj' and polarity=='positive':
                weakPos+=1
            if strength=='strongsubj' and polarity=='positive':
                strongPos+=1
            if strength=='weaksubj' and polarity=='negative':
                weakNeg+=1
            if strength=='strongsubj' and polarity=='negative':
                strongNeg+=1
            features['positivecount']=weakPos+(2*strongPos)
            features['negativecount']=weakNeg+(2*strongNeg)
    if 'positivecount' not in features:
        features['positivecount']=0
    if 'negativecount' not in features:
        features['negativecount']=0      
    return features

def liwc_features(document,word_features,poslist,neglist):
    d_words=set(document)
    features={}
    for w in word_features:
        features['contains({})'.format(w)]=(w in d_words)
    pos=0
    neg=0
    for w in d_words:
        if sentiment_read_LIWC_pos_neg_words.isPresent(w,poslist):
            pos+=1
        if sentiment_read_LIWC_pos_neg_words.isPresent(w,neglist):
            neg+=1
        features['positivecount']=pos
        features['negativecount']=neg
    if 'positivecount' not in features:
        features['positivecount']=0
    if 'negativecount' not in features:
        features['negativecount']=0  
    return features
  
def SL_liwc_features(document,word_features,SL,poslist,neglist):
    d_words=set(document)
    features={}
    for w in word_features:
        features['contains({})'.format(w)] = (w in d_words)
    weakPos=0
    strongPos=0
    weakNeg=0
    strongNeg = 0
    for w in d_words:
        if sentiment_read_LIWC_pos_neg_words.isPresent(w,poslist):
            strongPos+=1
        elif sentiment_read_LIWC_pos_neg_words.isPresent(w,neglist):
            strongNeg+=1
        elif w in SL:
            strength,posTag,isStemmed,polarity=SL[w]
            if strength=='weaksubj' and polarity=='positive':
                weakPos+=1
            if strength=='strongsubj' and polarity=='positive':
                strongPos+=1
            if strength=='weaksubj' and polarity=='negative':
                weakNeg+=1
            if strength=='strongsubj' and polarity=='negative':
                strongNeg+=1
        features['positivecount']=weakPos+(2*strongPos)
        features['negativecount']=weakNeg+(2*strongNeg)
    if 'positivecount' not in features:
        features['positivecount']=0
    if 'negativecount' not in features:
        features['negativecount']=0      
    return features
  
# Function to read kaggle training file, train, and test a classifier: 
def processkaggle(dirPath,limitStr):
    dirPath='/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews'
    limitStr=20000
    limit=limitStr
    os.chdir(dirPath)
    f=open('/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/corpus/train.tsv', 'r')
    # loop over lines in the file and use the first limit of them
    phrasedata=[]
    for line in f:
        # ignore the first line starting with 'Phrase'
        if (not line.startswith('Phrase')):
            #remove final end of line character
            line=line.strip()
            # each line has 4 items separated by tabs
            # ignore the phrase and sentence ids, keep the phrase and sentiment
            phrasedata.append(line.split('\t')[2:4])
    # pick a random length
    random.shuffle(phrasedata)
    phraselist=phrasedata[:limit]
    print('Read',len(phrasedata),'phrases, using',len(phraselist),'random phrases')
    print('All Phrases:')
    for phrase in phraselist[:10]:
        print(phrase)
    print(' ')
    wordtoken=[]
    processtoken=[]
    wordtoken=word_token(phraselist)
    processtoken=process_token(phraselist)
    print('Word Tokenized but without pre processing:')
    for phrase in wordtoken[:10]:
        print(phrase)
    print(' ')
    print('Word Tokenized and pre processed:')
    for phrase in processtoken[:10]:
        print(phrase)
    print(' ')
    
    # Filtering Tokens:
    filteredtokens=remove_characters(processtoken)
    unprocessedtokens=get_words(wordtoken)
  
    # Continue as usual to get all words and create word features
    unproc_word_features=bag_of_words(unprocessedtokens)
    word_features=bag_of_words(filteredtokens)
    
    # For Bigram Feature:
    unproc_bigramword_features=bag_of_words_bigram(unprocessedtokens,300)
    bigramword_features=bag_of_words_bigram(filteredtokens,300)
    
    
    # For Negation feature:
    negative_words=['abysmal','adverse','alarming','angry','annoy','anxious','apathy','appalling','atrocious','awful',
    'bad','banal','barbed','belligerent','bemoan','beneath','boring','broken',
    'callous','ca n\'t','clumsy','coarse','cold','cold-hearted','collapse','confused','contradictory','contrary','corrosive','corrupt','crazy','creepy','criminal','cruel','cry','cutting',
    'dead','decaying','damage','damaging','dastardly','deplorable','depressed','deprived','deformed''deny','despicable','detrimental','dirty','disease','disgusting','disheveled','dishonest','dishonorable','dismal','distress','do n\'t','dreadful','dreary',
    'enraged','eroding','evil','fail','faulty','fear','feeble','fight','filthy','foul','frighten','frightful',
    'gawky','ghastly','grave','greed','grim','grimace','gross','grotesque','gruesome','guilty',
    'haggard','hard','hard-hearted','harmful','hate','hideous','horrendous','horrible','hostile','hurt','hurtful',
    'icky','ignore','ignorant','ill','immature','imperfect','impossible','inane','inelegant','infernal','injure','injurious','insane','insidious','insipid',
    'jealous','junky','lose','lousy','lumpy','malicious','mean','menacing','messy','misshapen','missing','misunderstood','moan','moldy','monstrous',
    'naive','nasty','naughty','negate','negative','never','no','nobody','nondescript','nonsense','noxious',
    'objectionable','odious','offensive','old','oppressive',
    'pain','perturb','pessimistic','petty','plain','poisonous','poor','prejudice','questionable','quirky','quit',
    'reject','renege','repellant','reptilian','repulsive','repugnant','revenge','revolting','rocky','rotten','rude','ruthless',
    'sad','savage','scare','scary','scream','severe','shoddy','shocking','sick',
    'sickening','sinister','slimy','smelly','sobbing','sorry','spiteful','sticky','stinky','stormy','stressful','stuck','stupid','substandard','suspect','suspicious',
    'tense','terrible','terrifying','threatening',
    'ugly','undermine','unfair','unfavorable','unhappy','unhealthy','unjust','unlucky','unpleasant','upset','unsatisfactory',
    'unsightly','untoward','unwanted','unwelcome','unwholesome','unwieldy','unwise','upset','vice','vicious','vile','villainous','vindictive',
    'wary','weary','wicked','woeful','worthless','wound','yell','yucky',
    'are n\'t','cannot','ca n\'t','could n\'t','did n\'t','does n\'t','do n\'t','had n\'t','has n\'t','have n\'t','is n\'t','must n\'t','sha n\'t','should n\'t','was n\'t','were n\'t','would n\'t',
    'no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
    processnwords=negativeword_processing(negative_words)
    negative_words=negative_words+processnwords

    # For SL feature:
    SLpath='/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
    SL=sentiment_read_subjectivity.readSubjectivity(SLpath)
  
    # For LIWC feature:
    poslist,neglist=sentiment_read_LIWC_pos_neg_words.read_words()
    poslist=poslist+negativeword_processing(poslist)
    neglist=neglist+negativeword_processing(neglist)

    print("---------------------------------------------------")
    print("Top 15 Unprocessed tokens word features")
    print(unproc_word_features[:15])
    print("---------------------------------------------------")
    print("Top 15 Pre-processed tokens word features")
    print(word_features[:10])
    print("---------------------------------------------------")
    print("Top 15 Unprocessed tokens word features(Bigrams)")
    print(unproc_bigramword_features[:15])
    print("---------------------------------------------------")
    print("Top 15 Pre-processed tokens word features(Bigrams)")
    print(bigramword_features[:15])
    print("---------------------------------------------------")
  
    # Feature sets from a feature definition function (un-processed):
    unigramsets_without_preprocess=[(unigram_features(d,unproc_word_features),s) for (d,s) in wordtoken]
    print(" ")
    print("Unigramsets_without_preprocess: ")
    print(unigramsets_without_preprocess[0])
    saveFeaturesets.export_to_csv(unigramsets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/unigramsets_without_preprocess.csv")
    print(" ")

    bigramsets_without_preprocess=[(bigram_features(d,unproc_word_features,unproc_bigramword_features),s) for (d,s) in wordtoken]
    print("Bigramsets_without_preprocess: ")
    print(bigramsets_without_preprocess[0])
    saveFeaturesets.export_to_csv(bigramsets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/bigramsets_without_preprocess.csv")
    print(" ")
  
    negativesets_without_preprocess=[(negative_features(d,unproc_word_features,negative_words),s) for (d,s) in wordtoken]
    print("Negativesets_without_preprocess: ")
    print(negativesets_without_preprocess[0])
    saveFeaturesets.export_to_csv(negativesets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/negativesets_without_preprocess.csv")
    print(" ")
  
    possets_without_preprocess=[(POS_features(d,unproc_word_features),s) for (d,s) in wordtoken]
    print("POSsets_without_preprocess:")
    print(possets_without_preprocess[0])
    saveFeaturesets.export_to_csv(possets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/possets_without_preprocess.csv")
    print(" ")
  
    subjectivitysets_without_preprocess=[(SL_features(d,unproc_word_features,SL),s) for (d,s) in wordtoken]
    print("Subjectivitysets_without_preprocess:")
    print(subjectivitysets_without_preprocess[0])
    saveFeaturesets.export_to_csv(subjectivitysets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/subjectivitysets_without_preprocess.csv")
    print(" ")
  
    liwcsets_without_preprocess=[(liwc_features(d,unproc_word_features,poslist,neglist),s) for (d,s) in wordtoken]
    print("liwcsets_without_preprocess: ")
    print(liwcsets_without_preprocess[0])
    saveFeaturesets.export_to_csv(liwcsets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/liwcsets_without_preprocess.csv")
    print(" ")
  
    sl_liwcsets_without_preprocess=[(SL_liwc_features(d,unproc_word_features,SL,poslist,neglist),s) for (d,s) in wordtoken]
    print("SL_liwcsets_without_preprocess: ")
    print(sl_liwcsets_without_preprocess[0])
    saveFeaturesets.export_to_csv(sl_liwcsets_without_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/sl_liwcsets_without_preprocess.csv")
    print(" ")
  
    # feature sets from a feature definition function (Pre-processed):
    unigramsets_with_preprocess=[(unigram_features(d,word_features),s) for (d,s) in processtoken]
    print("Unigramsets_with_preprocess: ")
    print(unigramsets_with_preprocess[0])
    saveFeaturesets.export_to_csv(unigramsets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/unigramsets_with_preprocess.csv")
    print(" ")
  
    bigramsets_with_preprocess=[(bigram_features(d,word_features,bigramword_features),s) for (d,s) in processtoken]
    print("Bigramsets_with_preprocess: ")
    print(bigramsets_with_preprocess[0])
    saveFeaturesets.export_to_csv(bigramsets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/bigramsets_with_preprocess.csv")
    print(" ")
  
    negativesets_with_preprocess=[(negative_features(d,word_features,negative_words),s) for (d,s) in processtoken]
    print("Negativesets_with_preprocess: ")
    print(negativesets_with_preprocess[0])
    saveFeaturesets.export_to_csv(negativesets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/negativesets_with_preprocess.csv")
    print(" ")
    
    possets_with_preprocess=[(POS2_features(d,word_features),s) for (d,s) in wordtoken]
    print("POSsets_with_preprocess: ")
    print(possets_with_preprocess[0])
    saveFeaturesets.export_to_csv(possets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/negativesets_with_preprocess.csv")
    print(" ")

    subjectivitysets_with_preprocess=[(SL_features(d,word_features,SL),s) for (d,s) in processtoken]
    print("Subjectivitysets_with_preprocess: ")
    print(subjectivitysets_with_preprocess[0])
    saveFeaturesets.export_to_csv(subjectivitysets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/subjectivitysets_with_preprocess.csv")
    print(" ")

    liwcsets_with_preprocess=[(liwc_features(d,word_features,poslist,neglist),s) for (d,s) in processtoken]
    print("liwcsets_with_preprocess: ")
    print(liwcsets_with_preprocess[0])
    saveFeaturesets.export_to_csv(liwcsets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/liwcsets_with_preprocess.csv")
    print(" ")

    sl_liwcsets_with_preprocess=[(SL_liwc_features(d,word_features,SL,poslist,neglist),s) for (d,s) in processtoken]
    print("SL_liwcsets_with_preprocess: ")
    print(sl_liwcsets_with_preprocess[0])
    saveFeaturesets.export_to_csv(sl_liwcsets_with_preprocess,"/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/savedFeaturesets/sl_liwcsets_with_preprocess.csv")
    print(" ")
 
    #Accuracy Comparison - Naive Bayes vs Logistic Regression:
    print("Naive Bayes Classifier")
    print("---------------------------------------------------")
    print("Accuracy with Unigramsets_without_preprocess: ")
    nltk_naive_bayes(unigramsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with Bigramsets_without_preprocess: ")
    nltk_naive_bayes(bigramsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with negativesets_without_preprocess: ")
    nltk_naive_bayes(negativesets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with subjectivitysets_without_preprocess: ")
    nltk_naive_bayes(subjectivitysets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with liwcsets_without_preprocess: ")
    nltk_naive_bayes(liwcsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with sl_liwcsets_without_preprocess: ")
    nltk_naive_bayes(sl_liwcsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with POSsets_without_preprocess: ")
    nltk_naive_bayes(possets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("Accuracy with Unigramsets_with_preprocess: ")
    nltk_naive_bayes(unigramsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with Bigramsets_with_preprocess: ")
    nltk_naive_bayes(bigramsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with negativesets_with_preprocess: ")
    nltk_naive_bayes(negativesets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with subjectivitysets_with_preprocess: ")
    nltk_naive_bayes(subjectivitysets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with liwcsets_with_preprocess: ")
    nltk_naive_bayes(liwcsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with sl_liwcsets_with_preprocess: ")
    nltk_naive_bayes(sl_liwcsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with POSsets_with_preprocess: ")
    nltk_naive_bayes(possets_with_preprocess,0.1)  
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("Logistic Regression")
    print("---------------------------------------------------")
    print("Accuracy with Unigramsets_without_preprocess: ")
    sklearn_LReg(unigramsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with Bigramsets_without_preprocess: ")
    sklearn_LReg(bigramsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with negativesets_without_preprocess: ")
    sklearn_LReg(negativesets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with subjectivitysets_without_preprocess: ")
    sklearn_LReg(subjectivitysets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with liwcsets_without_preprocess: ")
    sklearn_LReg(liwcsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with sl_liwcsets_without_preprocess: ")
    sklearn_LReg(sl_liwcsets_without_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with POSsets_without_preprocess: ")
    sklearn_LReg(possets_without_preprocess,0.1)
    print("---------------------------------------------------") 
    print("---------------------------------------------------")
    print("Accuracy with Unigramsets_with_preprocess: ")
    sklearn_LReg(unigramsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with Bigramsets_with_preprocess: ")
    sklearn_LReg(bigramsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with negativesets_with_preprocess: ")
    sklearn_LReg(negativesets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with subjectivitysets_with_preprocess: ")
    sklearn_LReg(subjectivitysets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with liwcsets_with_preprocess: ")
    sklearn_LReg(liwcsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with sl_liwcsets_with_preprocess: ")
    sklearn_LReg(sl_liwcsets_with_preprocess,0.1)
    print("---------------------------------------------------")
    print("Accuracy with POSsets_with_preprocess: ")
    sklearn_LReg(possets_with_preprocess,0.1) 
  
    # Train classifier and show performance in cross-validation:
    labels=[0,1,2,3,4]
    num_folds=5
    
    # Without preprocessing:
    print("Unigramsets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,unigramsets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,unigramsets_without_preprocess,labels)
    print("Bigramsets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,bigramsets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,bigramsets_without_preprocess,labels)
    print("POSsets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,possets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,possets_without_preprocess,labels)
    print("Negativesets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,negativesets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,negativesets_without_preprocess,labels)
    print("Subjectivitysets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,subjectivitysets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,subjectivitysets_without_preprocess,labels)
    print("LIWCsets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,liwcsets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,liwcsets_without_preprocess,labels)
    print("SL+LIWC sets_without_preprocess:")
    Cross_Validation.naive_bayes(num_folds,sl_liwcsets_without_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,sl_liwcsets_without_preprocess,labels)
    
    # With preprocessing:
    print("Unigramsets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,unigramsets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,unigramsets_with_preprocess,labels)
    print("Bigramsets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,bigramsets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,bigramsets_with_preprocess,labels)
    print("POSsets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,possets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,possets_with_preprocess,labels)
    print("Negativesets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,negativesets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,negativesets_with_preprocess,labels)
    print("Subjectivitysets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,subjectivitysets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,subjectivitysets_with_preprocess,labels)
    print("LIWCsets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,liwcsets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,liwcsets_with_preprocess,labels)
    print("SL+LIWCsets_with_preprocess:")
    Cross_Validation.naive_bayes(num_folds,sl_liwcsets_with_preprocess,labels)
    Cross_Validation.logisticregression(num_folds,sl_liwcsets_with_preprocess,labels)

# Classifer Functions:
def nltk_naive_bayes(featuresets,percent):
    training_size=int(percent*len(featuresets))
    train_set,test_set=featuresets[training_size:],featuresets[:training_size]
    classifier=nltk.NaiveBayesClassifier.train(train_set)
    print("Naive Bayes Classifier")
    print("Accuracy: ",nltk.classify.accuracy(classifier,test_set))
    print("Showing most informative features: ")
    print(classifier.show_most_informative_features(10))
    confusionmatrix(classifier,test_set)
    print(" ")

def sklearn_LReg(featuresets,percent):
    training_size=int(percent*len(featuresets))
    train_set,test_set=featuresets[training_size:],featuresets[:training_size]
    classifier1=SklearnClassifier(LogisticRegression())
    classifier1.train(train_set)
    print("Logistic Regression")
    print("Accuracy: ",nltk.classify.accuracy(classifier1,test_set))
    print(" ")

#Function that returns mean accuracy, precision, recall, and F-measure of Cross-Validation:
def cross_validation_accuracy(num_folds,featuresets):
    subset_size=int(len(featuresets)/num_folds)
    accuracy_list=[]
    pos_precision_list=[]
    pos_recall_list=[]
    pos_fmeasure_list=[]
    neg_precision_list=[]
    neg_recall_list=[]
    neg_fmeasure_list=[]
    for i in range(num_folds):
        test_this_round=featuresets[(i*subset_size):][:subset_size]
        train_this_round=featuresets[:(i*subset_size)]+featuresets[((i+1)*subset_size):]
        classifier=nltk.NaiveBayesClassifier.train(train_this_round)
        accuracy_this_round=nltk.classify.accuracy(classifier,test_this_round)
        print(i,accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
        refsets=collections.defaultdict(set)
        testsets=collections.defaultdict(set)
        for i, (feats,label) in enumerate(testfeats):
            refsets[label].add(i)
            observed=classifier.classify(feats)
            testsets[observed].add(i)
        cv_accuracy=nltk.classify.util.accuracy(classifier,testing_this_round)
        cv_pos_precision=nltk.metrics.precision(refsets['pos'],testsets['pos'])
        cv_pos_recall=nltk.metrics.recall(refsets['pos'],testsets['pos'])
        cv_pos_fmeasure=nltk.metrics.f_measure(refsets['pos'],testsets['pos'])
        cv_neg_precision=nltk.metrics.precision(refsets['neg'],testsets['neg'])
        cv_neg_recall=nltk.metrics.recall(refsets['neg'],testsets['neg'])
        cv_neg_fmeasure=nltk.metrics.f_measure(refsets['neg'],testsets['neg'])
        pos_precision_list.append(cv_pos_precision)
        pos_recall_list.append(cv_pos_recall)
        neg_precision_list.append(cv_neg_precision)
        neg_recall_list.append(cv_neg_recall)
        pos_fmeasure_list.append(cv_pos_fmeasure)
        neg_fmeasure_list.append(cv_neg_fmeasure)
    print('Mean Accuracy: ',sum(accuracy_list)/num_folds)
    print('Precision: ',(sum(pos_precision_list)/n+sum(neg_precision_list)/n)/2)
    print('Recall: ',(sum(pos_recall_list)/n+sum(neg_recall_list)/n)/2)
    print('F-measure: ',(sum(pos_fmeasure_list)/n+sum(neg_fmeasure_list)/n)/2)

# Word Tokenizer Functions:
def word_token(phraselist):
    phrasedocs=[]
    for phrase in phraselist:
        tokens=nltk.word_tokenize(phrase[0])
        phrasedocs.append((tokens,int(phrase[1])))
    return phrasedocs

def process_token(phraselist):
    phrasedocs2=[]
    for phrase in phraselist:
        tokens=nltk.word_tokenize(phrase[0])
        tokens=lower_case(tokens)
        tokens=clean_text(tokens)
        tokens=remove_punctuation(tokens)
        tokens=remove_stopwords(tokens)
        tokens=stemmer(tokens)
        tokens=lemmatizer(tokens)
        phrasedocs2.append((tokens,int(phrase[1])))
    return phrasedocs2

# Pre-Processing Functions:
def lower_case(document):
    return [w.lower() for w in document]
  
def clean_text(document):
    cleantext=[]
    for review_text in document:
        review_text=re.sub(r"it 's", "it is",review_text)
        review_text=re.sub(r"that 's", "that is",review_text)
        review_text=re.sub(r"\'s", "\'s",review_text)
        review_text=re.sub(r"\'ve", "have",review_text)
        review_text=re.sub(r"wo n't", "will not",review_text)
        review_text=re.sub(r"do n't", "do not",review_text)
        review_text=re.sub(r"ca n't", "can not",review_text)
        review_text=re.sub(r"sha n't", "shall not",review_text)
        review_text=re.sub(r"n\'t", "not",review_text)
        review_text=re.sub(r"\'re", "are",review_text)
        review_text=re.sub(r"\'d", "would",review_text)
        review_text=re.sub(r"\'ll", "will",review_text)
        cleantext.append(review_text)
    return cleantext

def remove_punctuation(document):
    punct_removed=[]
    for w in document:
        punctuation=re.compile(r'[-_.?!/\%@,":;\'{}<>~`\()|0-9]')
        word=punctuation.sub("",w)
        punct_removed.append(word)
    return punct_removed

def remove_stopwords(document):
    stopwords=nltk.corpus.stopwords.words('english')
    newStopwords=[w for w in stopwords if w not in ['not', 'no', 'can','has','have','had','must','shan','do', 'should','was','were','won','are','cannot','does','ain', 'could', 'did', 'is', 'might', 'need', 'would']]
    return [w for w in document if not w in newStopwords]
  
def lemmatizer(document):
    wnl=nltk.WordNetLemmatizer() 
    lemma=[wnl.lemmatize(t) for t in document] 
    return lemma

def stemmer(document):
    p=nltk.PorterStemmer()
    stem=[p.stem(t) for t in document] 
    return stem
  
def negativeword_processing(negativewords):
    nwords=[]
    nwords=clean_text(negativewords)
    nwords=lemmatizer(nwords)
    nwords=stemmer(nwords)
    return nwords

def word_processing(word):
    wnl=nltk.WordNetLemmatizer()
    p=nltk.PorterStemmer()
    nwords=wnl.lemmatize(word)
    nwords=p.stem(nwords)
    return nwords
  
# Filtering Functions:
def remove_characters(document):
    word_list=[]
    for (w,label) in document:
        filtered_words=[x for x in w if len(x)>2]
        word_list.extend(filtered_words)
    return word_list

def get_words(document):
    word_list=[]
    for (w,sent) in document:
        word_list.extend(w)
    return word_list

def confusionmatrix(classifier_type,test_set):
    reflist=[]
    testlist=[]
    for (f,label) in test_set:
        reflist.append(label)
        testlist.append(classifier_type.classify(f))
    print("Confusion matrix: ")
    c_matrix=ConfusionMatrix(reflist,testlist)
    print(c_matrix)

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.
"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])


# In[ ]:




