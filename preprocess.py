from constants import BLANK, DATA, PATH
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

import string
import re


def clean_data(text):
    text = str(text)
    # remove punctuation
    punct = string.punctuation.replace("'", "")
    # every punct will be replaced by a space
    trantab = str.maketrans(punct, len(punct)*' ')
    text = text.translate(trantab)

    # remove digits
    text = re.sub('\d+', '', text)  # every digit is removed

    # lowercase
    text = text.lower()

    # remove stopwords
    stopwords_list = stopwords.words('english')
    # some words that might be indicative of sentiment
    whitelist = ["n't", "not", "no", BLANK]
    words = text.split()
    clean_words = [word for word in words if (
        word not in stopwords_list or word in whitelist) and len(word) > 1]
    text = ' '.join(clean_words)

    '''
    # stemming
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in words]
    text = ' '.join(stemmed_words)
    '''

    return text


def clean_labels(text):
    # lowercase
    text = text.lower()

    # remove punctuation
    punct = string.punctuation.replace("'", "")
    # every punct will be replaced by a space
    trantab = str.maketrans(punct, len(punct)*' ')
    text = text.translate(trantab)

    # replace space with '-'
    text = text.replace(' ', '-')

    return text

# iteratively call clean on data


def clean(text_list, type):
    clean_text_list = []
    if type == DATA:
        for text in text_list:
            clean_text_list.append(clean_data(text))
    else:
        for text in text_list:
            clean_text_list.append(clean_labels(text))
    return clean_text_list


# check if text is nan


def filter(data):
    filtered_data = []
    for data_item in data:
        if type(data_item) is not str or data_item == 'nan':
            filtered_data.append(BLANK)
        else:
            filtered_data.append(data_item)
    return filtered_data

# creating train-test split

def split(X, y, test_split):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_split, stratify=y)
    return X_train, y_train, X_test, y_test


def format_data(train_data, label, test_split):
    y = train_data.pop(label)

    X_train, y_train, X_test, y_test = split(train_data, y, test_split)
    X_train, y_train, X_dev, y_dev = split(X_train, y_train, test_split)

    X_train.insert(0, label, '__label__' + y_train)
    X_dev.insert(0, label, '__label__' + y_dev)
    X_test.insert(0, label, '__label__' + y_test)
    
    X_test.to_csv(PATH + 'test.csv', sep='\t', index=False, header=False)
    X_dev.to_csv(PATH + 'dev.csv', sep='\t', index=False, header=False)
    X_train.to_csv(PATH + 'train.csv', sep='\t', index=False, header=False)

    return 'test.csv', 'dev.csv', 'train.csv'
