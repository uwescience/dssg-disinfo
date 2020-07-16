import re
import pandas as pd

def find_url(text):
    '''
    Returns a list of all URLS embedded in a text 
    
    '''

    urls = []
    urls = (re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    return urls

    
def url_list(all_data):
    '''
    identifies urls in each article text and creates a new column with 
    the list of identified urls as well as 
    '''
    all_data['urls'] = [find_url(cell) for cell in all_data['article_text']]
    return 

def replace_url(all_data):
    '''
    Finds and replaces url with 'EMBD_URL' within article text
    '''
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    all_data['article_text'].replace(to_replace = pattern, value = 'EMBD_HTML', regex=True, inplace=True)
    return

def url_len(all_data): 
    '''
    counts embedded urls for a given article
    and records the number in a new column
    '''
    all_data['no_urls'] = [len(x) for x in all_data['urls']]
    return 

def process_urls(all_data):
    '''
    Pulling everything together
    and returns a processed dataframe
    '''
    url_list(all_data)
    url_len(all_data)
    replace_url(all_data)
    return all_data

