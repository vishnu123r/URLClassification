"""
This script preprocesses data for training the classidfier
"""


import pickle
from nltk.corpus import stopwords
import re

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

feature_text = list(data['URL'].values)

ret_text_lst = []

for t in feature_text:
    if type(t) != str:
         t = t.decode("UTF-8").encode('ascii','ignore')
         
    t = re.sub(r'[^a-zA-Z]',r' ',t)

    del_words = ['www','http','com','co','uk','org','https']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    text = (i.strip() for i in t.split())
    text = [t for t in text if t not in stop_words]
    text = " ".join(text)
    
    ret_text_lst.append(text)