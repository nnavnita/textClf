from constants import BLANK, FASTTEXT
from flair.data import Sentence
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
from tqdm import tqdm

import pandas as pd


def train(path, test, dev, train):
    corpus = NLPTaskDataFetcher.load_classification_corpus(
        Path(path), test_file=test, dev_file=dev, train_file=train)
    #word_embeddings = [WordEmbeddings(FASTTEXT), BertEmbeddings(BERT)]
    word_embeddings = [WordEmbeddings(FASTTEXT)]
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings, hidden_size=32, reproject_words=True, reproject_words_dimension=256)
    classifier = TextClassifier(
        document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train(path, save_final_model=False, train_with_dev=True, param_selection_mode=True)
    return path+'best-model.pt'


def predict(model, data, feature, label):
    classifier = TextClassifier.load(model)
    predictions = [[feature, label]]
    for value in tqdm(data):
        sentence = Sentence(value, True)
        classifier.predict(sentence, multi_class_prob=True)
        try:
            preds = []
            for label in sentence.labels:
                label = str(label).split('(')
                x = label[0].strip()
                y = float(label[1].replace('(', '').replace(')', ''))
                preds.append([x, y])
            preds = sorted(preds, key=lambda x: (x[1]), reverse=True)[:2]
            pred = [value]
            for x, y in preds:
                pred.append(x.replace('-', ' ').upper())
        except:  # if no prediction made
            pred = [value, BLANK]
        predictions.append(pred)
    return pd.DataFrame(predictions)


def evaluate(model, data, feature, label):
    true_labels = [value.replace('-', ' ').strip() for value in data[label]]
    predicted_labels = []
    classifier = TextClassifier.load(model)
    for value in tqdm(data[feature]):
        sentence = Sentence(value)
        classifier.predict(sentence, multi_class_prob=True)
        try:
            preds = []
            for label in sentence.labels:
                label = str(label).split('(')
                x = label[0].strip()
                y = float(label[1].replace('(', '').replace(')', ''))
                preds.append([x, y])
            preds = sorted(preds, key=lambda x: (x[1]), reverse=True)[0]
            pred = ','.join([x.replace('-', ' ') for x, y in preds])
        except:  # if no prediction made
            pred = BLANK
        predicted_labels.append(pred)
    # simple accuracy count
    count = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            count += 1
    return count/len(data)
