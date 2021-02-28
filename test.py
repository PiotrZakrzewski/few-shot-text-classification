import sys
from flair.data import Sentence
from flair.models.text_classification_model import TARSClassifier

tars = TARSClassifier.load('resources/taggers/hn_moneytalk/final-model.pt')

while True:
    sent = input("Type in title to classify:")
    sentence = Sentence(sent)
    tars.predict(sentence)
    print(sentence)
