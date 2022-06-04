#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import nltk
from nltk.corpus import movie_reviews
import random

#Function to Export Featuresets as a CSV File:
def export_to_csv(featuresets,file_path):
    out_dir='/Users/jmike877/Desktop/NLP/FinalProjectData/kagglemoviereviews/SavedFeaturesets'
    file_path=os.path.join(out_dir,'FeatureSet1.csv')
    x=open(file_path,'w')
    f_names=featuresets[0][0].keys()
    f_name_line=''
    for name in f_names:
        name=name.replace(',','CM')
        name=name.replace('"','QU')
        name=name.replace("'",'DQ')
        f_name_line+=name+','
    f_name_line+='class'
    x.write(f_name_line)
    x.write('\n')
    for f in featuresets:
        f_line=''
        for key in f_names:
            try:
                f_line+= str(f[0].get(key,[])) + ','
            except KeyError:
                continue
        if f[1]==0:
            f_line+=str('strongly negative')
        elif f[1]==1:
            f_line+=str('slightly negative')
        elif f[1]==2:
            f_line+=str('neutral')
        elif f[1]==3:
            f_line+=str('slightly positive')
        elif f[1]==4:
            f_line+=str('strongly positive')
        x.write(f_line)
        x.write('\n')
    x.close()


# In[ ]:




