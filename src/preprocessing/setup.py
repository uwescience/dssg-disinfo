import pandas as pd

def load():
    """
    Loads negative and positive articles and concatanates in a new dataframe
    
    Returns:
    dataframe: concatanated
    """
    df_neg = pd.read_csv('/data/dssg-disinfo/negative_articles_v3.csv')
    df_pos = pd.read_csv('/data/dssg-disinfo/positive_articles_v3.csv')
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    # put_df in '/data/dssg-disinfo/articles_v3.csv'
    return df

def drop_duplicates(dataframe, column):
    """
    Removes duplicated rows in dataframe according to a column
    
    Returns:
    dataframe: without duplicated column
    """
    dataframe.drop_duplicates(subset = column, keep='first', inplace=True)
    return dataframe

def drop_nonenglanguage(dataframe):
    """
    
    """
    return dataframe