from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.data import Sentence
from flair.models.text_classification_model import TARSClassifier


# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

# training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
train = SentenceDataset(
    [
        Sentence('I eat pizza').add_label('food_or_drink', 'food'),
        Sentence('Hamburgers are great').add_label('food_or_drink', 'food'),
        Sentence('I love drinking tea').add_label('food_or_drink', 'drink'),
        Sentence('Beer is great').add_label('food_or_drink', 'drink')
    ])

# test dataset consisting of two sentences (1 labeled as "food" and 1 labeled as "drink")
test = SentenceDataset(
    [
        Sentence('I ordered pasta').add_label('food_or_drink', 'food'),
        Sentence('There was fresh juice').add_label('food_or_drink', 'drink')
    ])

# make a corpus with train and test split
corpus = Corpus(train=train, test=test)

from flair.trainers import ModelTrainer

# 1. load base TARS
tars = TARSClassifier.load('tars-base')

# 2. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task("FOOD_DRINK", label_dictionary=corpus.make_label_dictionary())

# 3. initialize the text classifier trainer with your corpus
trainer = ModelTrainer(tars, corpus)

# 4. train model
trainer.train(base_path='resources/taggers/food_drink', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=1, # small mini-batch size since corpus is tiny
              max_epochs=10, # terminate after 10 epochs
              train_with_dev=True,
              )

# 1. Load few-shot TARS model
tars = TARSClassifier.load('resources/taggers/food_drink/final-model.pt')

# 2. Prepare a test sentence
sentence = Sentence("I am so glad you like burritos")

# 3. Predict for food and drink
tars.predict(sentence)
print(sentence)
