import spacy
from spacy.lang.en import English
import pandas as pd





# get sentences
def get_sentences_count(doc):
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    return (len(sentences))
        #return(ent_text)
    
def count_sent_df():
    nlp = English() 
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    
    data1['sentences_count'] = data1.apply(lambda row: get_sentences_count(row['doc']), axis = 1)
    return df2