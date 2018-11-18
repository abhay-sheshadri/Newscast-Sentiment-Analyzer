import os
import nltk
import requests
import json
import pdb;

# Used for sorting the Penn News Corpus into three text files (one for each sentiment)
# Uses online api
# Please replce training data files with your own human classified sentiment files

cwd = os.getcwd()
prev_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)

def get_sentences_from_data():
    sentences = []
    for root, dirs, files in os.walk(cwd):
        for file in files:
            with open(os.path.join(root, file), 'r') as file:
                data = file.read()
                data.replace("\t", "")
                data.replace("  ", ". ")
                for sent in nltk.tokenize.sent_tokenize(data):
                    sentences.append(sent)
    print(len(sentences))
    return sentences

def get_api_sentiment(text):
    url = "http://text-processing.com/api/sentiment/"
    payload = {"text": text}
    response = requests.post(url, data=payload)
    try:
        return response.json()["label"]
    except:
        print(response.status_code)
        pdb.set_trace()

def main():
    sentences = get_sentences_from_data()
    for sentence in sentences:
        sent = get_api_sentiment(sentence)
        with open(r"{0}/{1}.txt".format(prev_dir, sent), "a") as file:
            file.write(sentence + "\n")

if __name__ == "__main__":
    main()
