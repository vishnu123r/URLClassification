"""
This script preprocesses data for classification
"""

import pickle
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess():
    
    with open('feature.pkl','rb') as f:
        feature = pickle.load(f)
        
    with open('label.pkl','rb') as f:
        label = pickle.load(f)
        
    features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(feature, label, test_size=0.5, random_state=42)
    
    vectorizer = TfidfVectorizer()
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)
    
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
    
    
