import utils
import nltk
import math
import pickle

# Number of most occuring words used
max = 5000

def main():
    sentences = utils.get_sentences_from_data()
    words = []
    # Get all words in our dataset
    for category in sentences:
        for sentence in sentences[category]:
            words += utils.get_words_from_sentence(sentence)
    # Calculate frequency of words in dataset and create features for training
    dist = nltk.FreqDist(words)
    word_features = list(dist.keys())[:max]
    feature_sets = [(utils.find_features(utils.get_words_from_sentence(sentence), word_features), category)
                    for category in sentences
                    for sentence in sentences[category]]
    # Split features into training and testing sets
    set_length = math.floor(len(feature_sets) / 2)
    training_set = feature_sets[:set_length]
    testing_set = feature_sets[set_length:]
    # Train Naive Bayes algorithm and print accuracy
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Accuracy: {}".format(nltk.classify.accuracy(classifier, testing_set)))
    # Save classifer to file
    save_classifier = open(r"training/classifier.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

if __name__ == '__main__':
    main()
