from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


class Classifier(ClassifierI):

    def __init__(self):
        # Classifier list
        self._classifiers = [SklearnClassifier(MultinomialNB()),
                             SklearnClassifier(BernoulliNB()),
                             SklearnClassifier(LinearSVC()),
                             SklearnClassifier(LogisticRegression()),
                             SklearnClassifier(SGDClassifier())]

    def train(self, training_set):
        # Train all sci-kit classifiers
        for i in range(len(self._classifiers)):
            self._classifiers[i].train(training_set)
        # NLTK native naive bayes classifier is added here
        self._classifiers.append(NaiveBayesClassifier.train(training_set))

    def count_votes(self, votes):
        positive = votes.count("pos")
        negative = votes.count("neg")
        neutral = votes.count("neutral")
        # Positive or negative state if more votes than the other
        # and equal to or greater than neutral
        if positive > negative and positive >= neutral:
            return "pos"
        elif negative > positive and negative >= neutral:
            return "neg"
        # Neutral if greatest number of votes or positive and negative are equal
        else:
            return "neutral"

    # Basic NLTK Classifier functions
    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.classify(features))
        return self.count_votes(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            votes.append(classifier.classify(features))
        choice_votes = votes.count(self.count_votes(votes))
        conf_value = choice_votes / len(votes)
        return conf_value
