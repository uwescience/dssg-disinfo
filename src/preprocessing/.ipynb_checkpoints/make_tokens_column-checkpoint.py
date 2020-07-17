# Tokenize module

from nltk.tokenize import RegexpTokenizer

def tokenize_by_column(dataframe,column):
    """
        input: text from a particular column in dataframe
        
        return: dataframe with new column of tokenized text extracted from the input column 
    """
    tokenizer = RegexpTokenizer(r'\w+')
    dataframe['tokenized'+'_'+column] = [tokenizer.tokenize(str(col)) for col in dataframe[column]]
    return
