import ast
import pandas as pd
from sklearn.metrics import accuracy_score
from pyabsa import APCCheckpointManager


class PyABSAModel():
    def __init__(self):
        self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='multilingual',
                                                                             auto_device=True)
         
    def res_to_pyapsa(self, df: pd.DataFrame) -> pd.DataFrame:
        dict_pyabsa = {'text': [], 'term': [], 'asp': [], 'pol': [], 'start': [], 'end': []}
        for ind in df.index:
            txt = df['text'][ind]
            ss = ast.literal_eval(df['starts'][ind])
            es = ast.literal_eval(df['ends'][ind])
            cs = ast.literal_eval(df['cats'][ind])
            ps = ast.literal_eval(df['pols'][ind])
            for four in zip(ss, es, cs, ps):
                new_txt = f'{txt[:four[0]]}[ASP]{four[2]}[ASP]{txt[four[1]:]}'
                dict_pyabsa['text'].append(new_txt)
                dict_pyabsa['term'].append(txt[four[0]:four[1]])
                dict_pyabsa['asp'].append(four[2])
                dict_pyabsa['pol'].append(four[3])
                dict_pyabsa['start'].append(four[0])
                dict_pyabsa['end'].append(four[1])
        df_pyabsa = pd.DataFrame(dict_pyabsa)
        return df_pyabsa
    
    def predict_pyabsa(self, df_pyabsa: pd.DataFrame) -> pd.DataFrame:
        dict_pred_pyabsa = {'text': [], 'term': [], 'asp': [], 'pol': [], 'start': [], 'end': []}
        for ind in df_pyabsa.index:
            result = self.sent_classifier.infer(df_pyabsa['text'][ind], print_result=False)
            dict_pred_pyabsa['text'].append(df_pyabsa['text'][ind])
            dict_pred_pyabsa['term'].append(df_pyabsa['term'][ind])
            dict_pred_pyabsa['asp'].append(result['aspect'][0])
            dict_pred_pyabsa['pol'].append(result['sentiment'][0].lower())
            dict_pred_pyabsa['start'].append(df_pyabsa['start'][ind])
            dict_pred_pyabsa['end'].append(df_pyabsa['end'][ind])
        df_pred_pyabsa = pd.DataFrame(dict_pred_pyabsa)
        return df_pred_pyabsa
    
    def predict(self, data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        df_dev_pyabsa = self.res_to_pyapsa(data)
        self.df_pred = self.predict_pyabsa(df_dev_pyabsa)
        acc = accuracy_score(df_dev_pyabsa['pol'].to_list(), self.df_pred['pol'].to_list())
        return self.df_pred, acc
    
    def apply_to_res(self, df_res: pd.DataFrame) -> pd.DataFrame:
        for ind in df_res.index:
            start = df_res['span_start'][ind]
            stop = df_res['span_end'][ind]
            category = df_res['category'][ind]
            idex = self.df_pred[(self.df_pred['start'] == start) 
                                & (self.df_pred['end'] == stop) 
                                & (self.df_pred['asp'] == category)
                               ].index.tolist()
            if idex:
                df_res['sentiment'][ind] = self.df_pred['pol'][idex[0]]
        df_res = df_res[['review_id', 'category', 'span', 'span_start', 'span_end', 'sentiment']]
        return df_res