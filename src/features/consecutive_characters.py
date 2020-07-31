import pandas as pd

#read and concatanate negative and positive articles
df = pd.read_csv('/data/dssg-disinfo/articles_v3.csv')

def get_consecutive_count(punc_mark,df,new_column):
    """
    This function counts the number of consecutive characters in a text document
    When calling it, replace punc_count with the target character and
    new_column with the column title to be appended to the dataframe
    """
    articles = df['article_text']
    count_list = []
    splited_articles = []
    for article in articles:
        article_word = article.split(' ')
        splited_articles.append(article_word)

    consecutive_count = 0
    for art in splited_articles:
        for word in art:
            x  = 0
            for i in word:
                if (len(word) > 1) and  (word[x] == word[x-1] and i == punc_mark):
                    consecutive_count = consecutive_count + 1
                    break
                x = x+1
        punc_count = consecutive_count
        count_list.append(punc_count)
        consecutive_count = 0
    df[new_column] = count_list
    return df

# getting the consecutive number of commas 
get_consecutive_count(",",df,"consecutive_commas")

# getting the consecutive number of question marks 
get_consecutive_count("?",df,"consecutive_question_marks")

# getting the consecutive number of full stops 
get_consecutive_count(".",df,"concescutive_full_stop")

print(df.head())