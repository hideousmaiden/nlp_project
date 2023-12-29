import ast
import pandas as pd
from sklearn.metrics import accuracy_score
from pyabsa import APCCheckpointManager


class PyABSAModel():
    def __init__(self):
        self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='multilingual',
                                                                             auto_device=True)
         
    def res_to_pyapsa(self, df):
        dict_pyabsa = {'text': [], 'term': [], 'asp': [], 'pol': []}
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
        df_pyabsa = pd.DataFrame(dict_pyabsa)
        return df_pyabsa
    
    def predict_pyabsa(self, df_pyabsa):
        dict_pred_pyabsa = {'text': [], 'term': [], 'asp': [], 'pol': [], 'conf': []}
        for ind in df_pyabsa.index:
            result = self.sent_classifier.infer(df_pyabsa['text'][ind], print_result=False)
            dict_pred_pyabsa['text'].append(df_pyabsa['text'][ind])
            dict_pred_pyabsa['term'].append(df_pyabsa['term'][ind])
            dict_pred_pyabsa['asp'].append(result['aspect'][0])
            dict_pred_pyabsa['pol'].append(result['sentiment'][0].lower())
            dict_pred_pyabsa['conf'].append(result['confidence'][0])
        df_pred_pyabsa = pd.DataFrame(dict_pred_pyabsa)
        return df_pred_pyabsa
    
    def predict(self, data):
        df_dev_pyabsa = self.res_to_pyapsa(data)
        df_pred = self.predict_pyabsa(df_dev_pyabsa)
        acc = accuracy_score(df_dev_pyabsa['pol'].to_list(), df_pred['pol'].to_list())
        return df_pred, acc