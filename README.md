# Newscast Sentiment Analyzer
Classifies the sentiment of a caption from the newscast into three categories: positive, negative, and neutral.  It uses a voting algorithm to
count the votes of several different classifiers: Basic Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Linear Support Vector,
Logisitic Regression, and Stochastic Regression. Each classifier votes on the sentiment of a peice of text, and a voting algortihm is used to 
decide the winner.

# Running the classifier
Make sure you have python installed.  Run `pip install -r requirements.txt` in your shell.
Replace the training data files in training/data with your own.  Then, run `python trainer.py` in your shell.  It should print the accuracy of the classifier as well.
In order to use the program, run `python run.py` in your shell.  You can get the sentiment of sentences.

# Results
Images of results can be found in the screenshots folder.  After trained by the penn news corpus dataset, the classifier had an 86% accuracy.
