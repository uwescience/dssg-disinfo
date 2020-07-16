import pandas as pd

def format_date():
    """
    change date into datetime type
    extract time from publish date
    add new column with year
    add new column with month
    add new column with week
    """
    # read and concatanate negative and positive articles
    df = pd.read_csv('/data/dssg-disinfo/articles_v3.csv')
    
    #converting date published to datetime data type
    df[["publish_date"]] = df[["publish_date"]].apply(pd.to_datetime)
    
    # extracting time from datetime and changing format to m-d-y
    df['year'] = df['publish_date'].dt.to_period('Y')
    df['month'] = df['publish_date'].dt.to_period('M')
    df['week'] = df['publish_date'].dt.to_period('W')
    df['month_day_year'] = df['publish_date'].dt.to_period('D').dt.strftime('%m-%d-%Y')
    
    return df
    
    
    
    
    
    
