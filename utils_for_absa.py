import pandas as pd

from sklearn.metrics import accuracy_score


def get_join(file_reviews: str, file_aspects: str) -> pd.DataFrame:
    df_dev_reviews = pd.read_csv(file_reviews, sep='\t', names=['review_id', 'text'])
    df_finetuning_preds = pd.read_csv(file_aspects, sep='\t', names=['review_id', 'category', 'span', 
                                                                                'span_start', 'span_end', 'sentiment', 
                                                                                'word'])
    df_join = df_finetuning_preds.join(df_dev_reviews.set_index('review_id'), on='review_id')
    return df_join.drop(columns=['word'])

def accuracy(file_true, file_pred):
    df_true = pd.read_csv(file_true, sep='\t')
    df_pred = pd.read_csv(file_pred, sep='\t')
    return accuracy_score(df_true.iloc[:,-1:], df_pred.iloc[:,-1:])