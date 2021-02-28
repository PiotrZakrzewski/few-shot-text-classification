from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.data import Sentence
from flair.models.text_classification_model import TARSClassifier


# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

label_name = "topic"
finance = "finance"
crypto = "crypto"
money = "money"
tech = "tech"


# training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
train = SentenceDataset(
    [
        Sentence('Are You Trading or Gambling?').add_label(label_name, finance),
        Sentence('Amazon capitalization reached trillion dollars').add_label(label_name, finance),
        Sentence('Finance dictionary: SPACs and IPOs').add_label(label_name, finance),
        Sentence('Developer salaries development').add_label(label_name, money),
        Sentence('My annual income as developer since 2008').add_label(label_name, money),
        Sentence('How to maximize your income as a developer').add_label(label_name, money),
        Sentence('Levels.fyi salary information in tech').add_label(label_name, money),
        Sentence('New version of ruby').add_label(label_name, tech),
        Sentence('Python: 30 years in').add_label(label_name, tech),
        Sentence('Things I learned developing D3 library for visualization').add_label(label_name, tech),
        Sentence('Bitcoin price most volatile since 2019').add_label(label_name, crypto),
        Sentence('Cryptocurrency mining consumes as much energy as some countries').add_label(label_name, crypto),
        Sentence('Bitcoin is a scam').add_label(label_name, crypto),
    ])

# test dataset consisting of two sentences (1 labeled as "food" and 1 labeled as "drink")
test = SentenceDataset(
    [
        Sentence('Coinbase S-1 filing').add_label(label_name, finance),
        Sentence('Lessons about trading').add_label(label_name, finance),
        Sentence('Logic programming in 2021').add_label(label_name, tech),
        Sentence('Future of web is HTML over websockets').add_label(label_name, tech),
        Sentence('How much do data engineers make').add_label(label_name, money),
        Sentence('Canadian tech salaries').add_label(label_name, money),
        Sentence('Vulnerability found in a popular crypto wallet').add_label(label_name, crypto),
        Sentence('Ethereum smart contracts are useless').add_label(label_name, crypto),
    ])

# make a corpus with train and test split
corpus = Corpus(train=train, test=test)

from flair.trainers import ModelTrainer

# 2. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task("HN_MONEYTALK", label_dictionary=corpus.make_label_dictionary())

# 3. initialize the text classifier trainer with your corpus
trainer = ModelTrainer(tars, corpus)

# 4. train model
trainer.train(base_path='resources/taggers/hn_moneytalk', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=1, # small mini-batch size since corpus is tiny
              max_epochs=10, # terminate after 10 epochs
              train_with_dev=True,
              )

# 1. Load few-shot TARS model
tars = TARSClassifier.load('resources/taggers/hn_moneytalk/final-model.pt')

# 2. Prepare a test sentence
sentence = Sentence("Tesla added Bitcoin to its assets portfolio")

tars.predict(sentence)
print(sentence)
