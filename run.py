import pickle
import classifier
import utils

# Just a loop where the text enetered is classified if it is not blank
def main():
    classifier_file = open(r"training/classifier.pickle", "rb")
    classifier = pickle.load(classifier_file)
    while True:
        sentence = input("Enter a sentence:").lower()
        if sentence != "":
            words = utils.get_words_from_sentence(sentence)
            features = dict([(word, True) for word in words])
            result = classifier.classify(features)
            confidence = classifier.confidence(features)
            print("Sentiment:", result)
            print("Confidence:", confidence)

if __name__ == '__main__':
    main()
