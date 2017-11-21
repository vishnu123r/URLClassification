from preprocess import preprocess
from time import time
from sklearn.naive_bayes import GaussianNB

features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
clf.fit(features_train, labels_train)

t0 = time()
accuracy = clf.score(features_test, labels_test)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
clf.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")