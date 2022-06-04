#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

#This Function Returns 2 Lists - words in positive emotion class and words in negative emotion class:
def read_words():
    poslist=[]
    neglist=[]
    flexicon=open('SentimentLexicons/liwcdic2007.dic',encoding='latin1')
    wordlines=[line.strip() for line in flexicon]
    for line in wordlines:
        if not line=='':
            items=line.split()
            word=items[0]
            classes=items[1:]
        for c in classes:
            if c=='126':
                poslist.append(word)
            if c=='127':
                neglist.append(word)
    return (poslist,neglist)

#Function that Returns True if the Word is on the List or Returns False if otherwise (prefix text):
def isPresent(word,emotionlist):
    isFound=False
    for emotionword in emotionlist:
        if not emotionword[-1]=='*':
            if word==emotionword:
                isFound=True
                break
        else:
            if word.startswith(emotionword[0:-1]):
                isFound=True
                break
    return isFound


# In[ ]:




