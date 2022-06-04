#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk

#Function that returns a dictionary of words with their subjectivity information:
def readSubjectivity(path):
    path='/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
    flexicon=open(path,'r')
    sldict={}
    for line in flexicon:
        fields=line.split()
        strength=fields[0].split("=")[1]
        word=fields[2].split("=")[1]
        posTag=fields[3].split("=")[1]
        stemmed=fields[4].split("=")[1]
        polarity=fields[5].split("=")[1]
        if (stemmed=='y'):
            isStemmed=True
        else:
            isStemmed=False
        sldict[word]=[strength,posTag,isStemmed,polarity]
    return sldict

#Function that returns 3 lists - words in positive, neutral, and negative subjectivity classes:
def read_subjectivity_three_types(path):
    path='/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
    poslist=[]
    neutrallist=[]
    neglist=[]
    flexicon=open(path,'r')
    wordlines=[line.strip() for line in flexicon]
    for line in wordlines:
        if not line=='':
            items=line.split()
            word=items[2][(items[2].find('=')+1):]
            polarity=items[5][(items[5].find('=')+1):]
            if polarity=='positive':
                poslist.append(word)
            if polarity=='neutral':
                neutrallist.append(word)
            if polarity=='negative':
                neglist.append(word)
    return (poslist,neutrallist,neglist)


# In[ ]:




