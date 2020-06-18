from __future__ import absolute_import, division, print_function, unicode_literals
import dill
import nltk
from nltk import pos_tag
import re
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

TOPIC = 'coronavirus'

def add_document_tags(corpus, label_type):
    """Summary or Description of the Function

    Parameters:
    corpus ():
    label_type ():

    Returns:
    list of document tags
    
   """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled

def get_wordnet_pos(treebank_tag):
    """Get part of speech from tag

    Parameters:
    treebank_tag (tag)

    Returns:
    wordnet part of speech
    
   """

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def filter_text(text, case_sensitive=False):
    """Filter out filler words and reduce words to lemmatized version, i.e.
    planned, planner, planning, plans --> plan

    Parameters:
    text (string): text to be filtered
    case_sensitive (bool): whether the tokens should be treated as case-sensitive
    with respect to eliminating stopwords

    Returns:
    string of reconnected lemmatized tokens with no stopwords
   """
    pattern = r'[^a-zA-z0-9\s]'
    try:
        text = re.sub(pattern, '', text)
    except:
        print(text)
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if case_sensitive:
        tokens = [token for token in tokens if token not in stopwords]
    else:
        tokens = [token for token in tokens if token.lower() not in stopwords]
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])) for token in tagged_tokens]
    return ' '.join(tokens)


def retrain_topic_frame():
    """MVP implementation of topic model

   """

    TRAIN_FILENAME_1 = 'pos/labeled_'+TOPIC+'_article_corpus.csv'
    TRAIN_FILENAME_0 = 'neg/'+TOPIC+'_article_corpus.csv'

    corpus = pd.read_csv(TRAIN_FILENAME_1)
    text = [filter_text(article) for article in corpus['text'].values]
    num_samples = len(text)
    pks = [pk for pk in corpus['pk'].values]
    labels = ['pos' for t in text]
    corpus = pd.read_csv(TRAIN_FILENAME_0)
    corpus = corpus.sample(n=num_samples)
    text += [filter_text(article) for article in corpus['text'].values]
    labels += ['neg' for t in range(num_samples)]
    pks += [pk for pk in corpus['pk'].values]

    df = pd.DataFrame({'pk':pks, 'text':text, 'label':labels})
    df = df.sample(frac=1)

    classifiers = [
    ("SGD", SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=42, max_iter=10, tol=None))
    ]

    heldout = [0.2]
    rounds = 50

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for _ in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(df['text'].values, df['label'].values, test_size=i, random_state=rng)
                count_vect = TfidfVectorizer(max_features=15000)
                count_vect.fit(X_train)
                X_train = count_vect.transform(X_train)
                X_test = count_vect.transform(X_test)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        print(TOPIC, yy)
    
    with open(TOPIC+'_{}_clf.pkl'.format(name), 'wb') as f:
        dill.dump(clf, f)
    with open(TOPIC+'_{}_count_vect.pkl'.format(name), 'wb') as f:
        dill.dump(count_vect, f)

if __name__ == '__main__': 
    retrain_topic_frame()
