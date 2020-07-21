# Tokenize module

from nltk.tokenize import RegexpTokenizer
import pandas as pd

def tokenize_by_column(dataframe,column):
    """
        input: text from a particular column in dataframe
        
        returns: new dataframe with old columns plus the new column of tokenized text extracted from the input column 
    """
    tokenizer = RegexpTokenizer(r'\w+')
    new_col = [tokenizer.tokenize(str(col)) for col in dataframe[column]]
    df_new_col = pd.DataFrame({'tokenized_' + column: new_col})
    new_df = dataframe.join(df_new_col)
    return new_df


