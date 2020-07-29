import spacy
from spacy.lang.en import English
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
import os


# Path to the environment variables file .env
env_path = '/data/dssg-disinfo/.env'



# get a list of text identified as entities
def get_entity_text(doc):
    ''' Performs named entity recognition for a given article text
    and returns a list of said entities. 
    
    Parameters
    ----------
    doc: doc object 
        Doc object for which you want the Named Entity Analysis to be performed
    Returns
    -------
    ent_text: list
        Returns a list containing names of all named entities in a given doc object
        
    '''
    ent_text = []
    for ent in doc.ents:
        ent_text.append(ent.text)
    return (ent_text)

# get a list of entity labels
    ''' Labels identified Named Entities in a given article text and 
    returns a list of said labels.  
    
    Parameters
    ----------
    doc: doc object 
        Doc object for which you want the Named Entity Analysis to be performed
    Returns
    -------
    ent_text: list
        Returns a list of labels for Named Entities recognized in a given text
        
    '''
def get_entity_labels(doc):
    ent_text = []
    for ent in doc.ents:
        ent_text.append(ent.label_)
    return (ent_text)


def generate_named_entities(DATA=None):
    ''' Identified Named Entities in the text columns of a dataframe
    and returns a new dataframe containing columns with Named Entities and 
    Labels for those Named Entities
    
    Parameters
    ----------
    df: DataFrame
        DataFrame for which you wish to do Named Entity Analysis. If left blank, 
        defaults to CLEAN_DATA dataset. 
        
    Returns
    -------
    dataframe
        Returns DataFrame with article_pk, label and named entity columns 
        
    '''
    PATH = os.getenv("PATH") # Path to the dataframe DATA
    
    if DATA is None:
        DATA_NAME = os.getenv("CLEAN_DATA")
    else:
        DATA_NAME = DATA
        
    df = pd.read_csv(os.path.join(PATH,DATA_NAME)) # Load DATA_NAME from PATH
    
    
    nlp = spacy.load("en_core_web_sm") # Load English multi-task CNN trained on OntoNotes. Assigns context-specific token vectors, POS tags, dependency parse and named entities.
    
    df['ent'] = df.apply(lambda row: get_entity_text(row['doc']), axis = 1)
    df['ent_label'] = df.apply(lambda row: get_entity_labels(row['doc']), axis = 1)
    df2=df[['article_pk','label','ent','ent_label']].copy()
    
    return df2