import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])

neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])

training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
test = pos[int((.9)*len(pos)):] + neg[int((.9)*len(neg)):]

print('len test',len(test))
print('len train', len(training))

from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()

example1 = "this workshop is no awesome."

print(classifier.classify(format_sentence(example1)))

example2 = "this workshop is awful."

print(classifier.classify(format_sentence(example2)))

from nltk.classify.util import accuracy
print(accuracy(classifier, test))
