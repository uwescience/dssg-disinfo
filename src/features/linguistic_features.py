import spacy
from collections import Counter
import pandas as pd
from spellchecker import SpellChecker


# GLOBAL VARIABLES
informal_contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',
                'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',
                'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',
                'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',
                'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',
                'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',
                'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',
                'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',
                'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',
                'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',
                'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',
                'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',
                'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',
                'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',
                'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',
                'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',
                'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',
                'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',
                'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']


nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()  


# Tagging parts of speech
def tag_pos(text):
    ''' Tags each word in a given text with part-of-speech(POS) tag and 
    returns a dictionary with pos/frequency
    
    Parameters
    ----------
    text: list
        List of tokens processed by SpaCy's nlp module
        
    Returns
    -------
    dictionary
        The dictionary word_freq with part-of-speech count in a given text


POS TAG Scheme 
        
POS	   Description	     Examples

ADJ	   adjective	     big, old, green, incomprehensible, first
ADP	   adposition	     in, to, during
ADV	   adverb	         very, tomorrow, down, where, there
AUX	   auxiliary	     is, has (done), will (do), should (do)
CONJ   conjunction       and, or, but
CCONJ  coordinating      and, or, but
       conjunction	
DET	   determiner	     a, an, the
INTJ   interjection      psst, ouch, bravo, hello
NOUN   noun	             girl, cat, tree, air, beauty
NUM	   numeral	         1, 2017, one, seventy-seven, IV, MMXIV
PART   particle	         ’s, not,
PRON   pronoun	         I, you, he, she, myself, themselves, somebody
PROPN  proper noun       Mary, John, London, NATO, HBO
PUNCT  punctuation	     ., (, ), ?
SCONJ  subordinating     if, while, that
       conjunction	
SYM	   symbol	        $, %, §, ©, +, −, ×, ÷, =, :), 😝
VERB   verb	            run, runs, running, eat, ate, eating
X	   other	        sfpksdpsxmsa
SPACE  space	    
    '''
    total = []
    for token in text:
        total.append(token.pos_)
    word_freq = Counter(total)
    return word_freq


def count_part_of_speech(df):
    ''' Performs part-of-speech(POS) identification on tokens in article_text
    for a give dataframe and creates a new dataframe with POS frequencies per article. 
    
    Parameters
    ----------
    df: DataFrame
        DataFrame for which you want to add part-of-speech count features
        
    Returns
    -------
    dataframe
        Returns DataFrame with article_pk as the index and part-of-speech
        frequencies/article as columns. 
        
    '''
    df['doc'] = df.apply(lambda row : nlp(row['article_text']) ,axis=1)
    df['doc2'] = df.apply(lambda row: tag_pos(row['doc']), axis=1)
    df2 = df[['article_pk','doc2']].copy()
    df3 = pd.concat([df2.drop(['doc2'], axis=1), df2['doc2'].apply(pd.Series)], axis=1)
    df3.fillna(value=0, inplace=True)
    return (df3)
    

# Checking for spelling

def check_for_spelling_errors(df):
    ''' Uses pyspellchecker 0.5.4 to identify misspelled tokens in each sample
    and returns number of identified misspellings as well as a list of misspelled words. 
    
    Parameters
    ----------
    df: DataFrame
        DataFrame for which you want to do a spell-check
        
    Returns
    -------
    df2: dataframe
        Returns DataFrame with spell-checked values columns and article_pk
        
    '''
    df['misspelled_list'] = df.apply(lambda row: list(spell.unknown(row['tokenized_article_text'])), axis=1)
    df['number_misspelled'] = df.apply(lambda row: len(row['misspelled_list']), axis=1)
    df2 = data[['article_pk','misspelled_list','number_misspelled']].copy()
    return df2 

# Identifying Contractions

def num_contract(text):
    text_list = text.split()
    num = len([word for word in text_list if word in informal_contractions])
    return num

def count_informal_contractions(df):
     ''' Checks number of infomral contractions in article text and 
     returns a column with their count. 
    
    Parameters
    ----------
    df: DataFrame
        DataFrame for which you want to check for presence of informal contractions
        
    Returns
    -------
    df2: dataframe
        Returns DataFrame with a new column counting counting informal contractions, article_pk column
        
    '''
    df['no_contractions'] = df.apply(lambda row: num_contract(row['article_text']), axis = 1)
    df2 = df[['article_pk','no_contractions']].copy()
    return df2

