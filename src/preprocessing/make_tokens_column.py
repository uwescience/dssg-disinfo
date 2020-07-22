# Tokenize module

from nltk.tokenize import RegexpTokenizer
<<<<<<< HEAD
import pandas as pd
=======
>>>>>>> 10fac11... pushing the code to process text data columns to tokenized text columns

def tokenize_by_column(dataframe,column):
    """
        input: text from a particular column in dataframe
        
<<<<<<< HEAD
        returns: new dataframe with old columns plus the new column of tokenized text extracted from the input column 
    """
    tokenizer = RegexpTokenizer(r'\w+')
    new_col = [tokenizer.tokenize(str(col)) for col in dataframe[column]]
    df_new_col = pd.DataFrame({'tokenized_' + column: new_col})
    new_df = dataframe.join(df_new_col)
    return new_df


=======
        return: dataframe with new column of tokenized text extracted from the input column 
    """
    tokenizer = RegexpTokenizer(r'\w+')
    dataframe['tokenized'+'_'+column] = [tokenizer.tokenize(str(col)) for col in dataframe[column]]
    return
>>>>>>> 10fac11... pushing the code to process text data columns to tokenized text columns
