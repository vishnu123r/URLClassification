import pandas as pd
import pickle
from nltk.corpus import stopwords
import re

"""
This script Labels the data and extracts features and labels for further processing
The features are extrcted to a list and saved as feature.pkl and labels are saved as label.pkl
"""



data = pd.read_table('classification.tsv', error_bad_lines = False)
data = data.loc[:,['Primary Category', 'URL']]

#Labelling the data
lab = set(data['Primary Category'].values)
lab = dict(enumerate(lab,1))
lab = dict (zip(lab.values(),lab.keys()))

label = list(map(lab.get, list(data['Primary Category'].values)))

data['label'] = pd.Series(label).values
data = data.loc[:, ['URL','label']]

with open('data.pkl','wb') as f:
    pickle.dump(data, f)
    
with open('label.pkl', 'wb') as f:
    pickle.dump(label, f)


#Parsing and cleaning URL(Features)    
feature_text = list(data['URL'].values)

features = []
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
    
    features.append(text)
    
with open('feature.pkl', 'wb') as f:
    pickle.dump(features, f)
    