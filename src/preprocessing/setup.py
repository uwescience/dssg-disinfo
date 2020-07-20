import pandas as pd
import os
from langdetect import detect

def load_cleandata():
    """
    Concatanates negative and positive articles
    Drops empty article_text rows
    Removes duplicated article_text
    Remove non-english article_text from the dataframe
    Remove noisy characters from article_text, article_headline
    Converting all characters in article_text, article_headline to ascii- removes emoticons
    Export clean data
    """
    PATH = '/data/dssg-disinfo/'
    
    # Concatanate negative and positive articles
    df_neg = pd.read_csv(PATH+'negative_articles_v3.csv')
    df_pos = pd.read_csv(PATH+'positive_articles_v3.csv')
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
    
    # Removing noisy characters from article-text
    df['article_text'] = [article.replace('\n', ' ') for article in df.article_text]
    df['article_headline'] =[headline.replace('\n', ' ') for headline in df.article_headline]
    
    # Replace anything other than alphabets(a-z,A-Z),?,!,whitespaces,0-9,comma, fullstops, dashes with a space
    df['article_text'] = [str(article).replace('[^a-zA-Z*|\?*|!*|\s*|0-9*|,*|.*|\-*]', ' ') for article in df.article_text]
    df['article_headline'] = [str(headline).replace('[^a-zA-Z*|\?*|!*|\s*|0-9*|,*|.*|\-*]', ' ') for headline in df.article_headline]
    
    # Converting all characters to ascii
    df['article_text'] = [article.str.encode('ascii', errors='ignore') for article in df.article_text]
    df['article_headline'] = [headline.str.encode('ascii', errors='ignore') for headline in df.article_headline]
    
    # Export clean data
    df.to_csv(PATH+'articles_v3.csv', index=False)
    return