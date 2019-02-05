#combining the algo for voting system

import nltk
import random
from nltk.corpus import movie_reviews 
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode



class VoteClassifer(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


documents = [(list(movie_reviews.words(fileid)), category)
			  for category in movie_reviews.categories()
			  for fileid in movie_reviews.fileids(category)]


#documents = []


#for category in movie_reviews.categories():
#	for fileid in movie_reviews.fileids(category):
#		documents.append(list(movie_reviews(fileid)), category)


random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occcurences * liklihood / edvidence

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set)))
classifier.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

#MultionmialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set)))

#GaussianNB

#gaussian_classifier = SklearnClassifier(GaussianNB())
#gaussian_classifier.train(training_set)
#print("GaussianNB_classifier Algo accuracy:", (nltk.classify.accuracy(gaussian_classifier, testing_set)))

# BernoulliNB

bernoull_classifier = SklearnClassifier(BernoulliNB())
bernoull_classifier.train(training_set)
print("BernoulliNB_classifier Algo accuracy:", (nltk.classify.accuracy(bernoull_classifier, testing_set)))

# LogisticRegression

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Algo accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

# SGDClassifier

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)))

# SVC

#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier Algo accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set)))

#LinearSVC,

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

# NuSVC

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Algo accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)))


voted_classifier = VoteClassifer(classifier, MNB_classifier,bernoull_classifier, LinearSVC_classifier, NuSVC_classifier, SGDClassifier_classifier, LogisticRegression_classifier)

print("voted_classifier Algo accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)))

print("classification:", voted_classifier.classify(testing_set[0][0]), "  confidence %:", voted_classifier.confidence(testing_set[0][0]))

print("classification:", voted_classifier.classify(testing_set[1][0]), "  confidence %:", voted_classifier.confidence(testing_set[1][0]))
print("classification:", voted_classifier.classify(testing_set[2][0]), "  confidence %:", voted_classifier.confidence(testing_set[2][0]))
print("classification:", voted_classifier.classify(testing_set[3][0]), "  confidence %:", voted_classifier.confidence(testing_set[3][0]))
print("classification:", voted_classifier.classify(testing_set[4][0]), "  confidence %:", voted_classifier.confidence(testing_set[4][0]))
print("classification:", voted_classifier.classify(testing_set[5][0]), "  confidence %:", voted_classifier.confidence(testing_set[5][0]))
print("classification:", voted_classifier.classify(testing_set[6][0]), "  confidence %:", voted_classifier.confidence(testing_set[6][0]))