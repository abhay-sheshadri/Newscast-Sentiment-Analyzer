import pickle
import classifier
import utils

def main():
    # Open classifier file and read bytes
    classifier_file = open(r"training/classifier.pickle", "rb")
    classifier = pickle.load(classifier_file)
    # Just a loop where the text enetered is classified if it is not blank
    while True:
        sentence = input("Enter a sentence:").lower()
        if sentence != "":
            words = utils.get_words_from_sentence(sentence)
            features = dict([(word, True) for word in words])
            result = classifier.classify(features)
            print("Sentiment:", result)

if __name__ == '__main__':
    main()
