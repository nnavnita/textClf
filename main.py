from constants import DATA, FEATURES, LABELS, PATH, PREDICT, RES_PATH, TRAIN
from preprocess import clean, filter, format_data, split
from model import evaluate, predict, train

import json
import pandas as pd
import sys
import time

start_time = time.time()
# read and preprocess data
mode = sys.argv[1]
input = pd.read_csv(sys.argv[2])
params = json.load(open(sys.argv[3]))

feature = params[FEATURES]
label = params[LABELS]
print('FEATURE: {}\nLABEL: {}'.format(feature, label))

data = pd.DataFrame(columns={label, feature})
data[feature] = filter(input[feature])
if mode == TRAIN:
    # preprocess labels
    data[label] = filter(input[label])
    data[label] = clean(data[label], 'label')
    y = data.pop(label)
    X_train, y_train, X_test, y_test = split(data, y, 0.3) #train-test ratio 70:30

    X_train[label] = y_train
    X_test[label] = y_test

    X_test.to_csv(PATH+'test_file.csv', sep=',')

    print('Begin training -- TRAIN: {} TEST: {}'.format(len(X_train),
                                                        len(X_test)))
    # create the datasets for training
    test_file, dev_file, train_file = format_data(X_train, label, 0.3)

    # train model
    model = train(PATH, test_file, dev_file, train_file)

    # evaluate
    score = evaluate(model, X_test, feature, label)
    print('ACC SCORE: TOP 1 = {}'.format(score))

elif mode == PREDICT:
    model = PATH + 'best-model.pt'
    path = RES_PATH + label + '.csv'
    predictions = predict(model, data[feature], feature, label)
    predictions.to_csv(path, sep=',')
    print('SAVED TO: {}'.format(path))

print('TIME TAKEN: {}s'.format(time.time() - start_time))
