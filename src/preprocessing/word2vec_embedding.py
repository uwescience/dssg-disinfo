import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import string as str
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import glob
import os


def w2v_embedding(src_file_path, bin_file_destination_path, txt_file_destination_path):
    
    """
    This function converts words in a text file into word embeddings using word2vec
    It accepts input through src_file_path parameter
    Then converts the words to embeddings and save then in bin_file_destination_path
    Finally, the document is converted into a text file and saved in txt_file_destination_path
    """
    
    df = open(file_path, "r")
    df = df.read()
    tokens1 = word_tokenize(df)
    
    token_list = []
    
    models = KeyedVectors.load_word2vec_format(bin_destination_file_path, binary=True)
    
    models.save_word2vec_format(txt_file_destination_path, binary=False)
    
    return txt_file_destination_path