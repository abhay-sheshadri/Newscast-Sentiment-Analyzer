import os
import nltk


# Reads all the files containing sentences for each sentiment
def get_sentences_from_data():
    sentences = {"neg": [], "pos": [], "neutral": [],}
    for sentiment in sentences:
        with open(r"training/data/{}.txt".format(sentiment)) as file:
            sentences[sentiment] = file.readlines()
            # Just for cleaning up the data
            sentences[sentiment] = [value for value in sentences[sentiment]
                                    if value != '\t']
    return sentences

# Given a sentence, returns all words in the sentence that weren't stop words or titles
def get_words_from_sentence(sent):
    words = []
    for word in nltk.tokenize.word_tokenize(sent):
        if word not in nltk.corpus.stopwords.words("english"):
            words.append(word.lower().replace('\n', ''))
    return words

# Returns dictionary of features
def find_features(words, word_features):
    words = set(words)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features
