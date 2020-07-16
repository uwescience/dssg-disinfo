import pandas as pd
from langdetect import detect

def load_cleandata():
    """
    Concatanates negative and positive articles
    Drops empty article_text rows
    Removes duplicated article_text
    Remove non-english article_text from the dataframe
    Export clean data
    """
    # Concatanate negative and positive articles
    df_neg = pd.read_csv('/data/dssg-disinfo/negative_articles_v3.csv')
    df_pos = pd.read_csv('/data/dssg-disinfo/positive_articles_v3.csv')
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Drop empty article_text rows
    df.dropna(subset=['article_text'], inplace=True)
    
    # Drop duplicated article_text
    df.drop_duplicates(subset = 'article_text', keep='first', inplace=True)
    
    # Index of non-english rows
    non_en_index = []
    for index, row in df.iterrows():
        # Explicitly converting article_text to string because a few of the rows were being captured as non-strings
        lang = detect(str(row['article_text']))
        if lang != 'en':
            non_en_index.append(index)

    # Removing non-english articles        
    df.drop(non_en_index, inplace= True)
    
    # Export clean data
    df.to_csv('/data/dssg-disinfo/articles_v3.csv', index=False)
    return