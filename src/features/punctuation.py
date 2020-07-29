import pandas as pd
from string import punctuation
from collections import Counter

#read and concatanate negative and positive articles
df = pd.read_csv('/data/dssg-disinfo/articles_v3.csv')

# getting the number of punctuation
def get_count(punc_mark, df):
    
    punc_count = df.article_text.str.count(punc_mark)
    
    return punc_count

# getting number of commas
df['comma_count'] = get_count(",", df)

# getting number of full stops
df['full_stop_count'] = get_count('.', df)

# getting number of exclaimation marks
df['mark_count'] = get_count("!", df)

# getting the ratio of punctuation
def get_ratio(punc_mark, df):
    
    total_count = df.full_stop_count + df.comma_count + df.mark_count
    
    punc_ratio = punc_mark/total_count
    
    return punc_ratio

# getting ratio of commas
df['comma_ratio'] = get_ratio(df.comma_count, df)

# getting ratio of full stops
df['stop_ratio'] = get_ratio(df.full_stop_count, df)

# getting ratio of exclamaition marks
df['mark_ratio'] = get_ratio(df.mark_count, df)

# getting average of a punctuation mark
def get_average(punc_count, df):
    article_length = len(df.article_text)
    
    punc_average = article_length/punc_count
    
    return punc_average

# getting average commas
df['avg_comma'] = get_average(df.comma_count, df)

# getting average full stops
df['avg_stop'] = get_average(df.full_stop_count, df)

# getting average exclaimation marks
df['avg_mark'] = get_average(df.mark_count, df)

print(df.head)