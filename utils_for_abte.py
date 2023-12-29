import time
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer as twt
from collections import defaultdict
import torch

def clean_data(review_path, aspect_path):
    with open(review_path, 'r', encoding='utf-8') as f:
        texts = f.read().split('\n')
    reviews = dict()
    for k in texts:
        try:
            text_n, text_b = k.split('\t')
            reviews[text_n] = text_b
        except:
            print(k)
    aspect_cats = {}
    indd = 0
    with open(aspect_path, 'r') as fg:
        for line in fg:
            line = line.rstrip('\r\n').split('\t')
            if line[0] not in aspect_cats:
                aspect_cats[line[0]] = {"starts":[], "ends":[], "cats":[], "pols":[]}
            aspect_cats[line[0]]["starts"].append(int(line[3]))
            aspect_cats[line[0]]["ends"].append(int(line[4]))
            aspect_cats[line[0]]["cats"].append(line[1])
            aspect_cats[line[0]]["pols"].append(line[5])
            aspect_cats[line[0]]["text"] = reviews[line[0]]
    return pd.DataFrame(aspect_cats).transpose()

# номера текстов не нужны самой модели но потом понадобятся при предикте чтобы записать результаты
# поэтому они не идут в датасет модели а выделяются потом отдельно
def clean_idx(aspect_path):
  idx = []
  with open(aspect_path, 'r') as fg:
        for line in fg:
            line = line.rstrip('\r\n').split('\t')
            if line[0] not in idx:
                idx.append(line[0])
        return idx
		
#пройтись по текстам, по токенам в них и выбрать те токены, которые == или внутри промежутка выделенных аспектов. замаркировать эти токены
def spans_to_tokens(starts, ends, cats, pols, text):
    cats_to_ids = {'Food':1, 'Interior':2,'Price':3,'Service':4,'Whole':5}
    pols_to_ids = {'positive':1, 'negative':2, 'neutral':3, 'both':4}
    tokens = list(twt().span_tokenize(text))
    real_tokens = list(twt().tokenize(text))
    pairs = list(zip(starts, ends))
    token_cats = []
    token_pols = []
    for token in tokens:
        found = False
        for pair, cat, pol in zip(pairs, cats, pols):
            if token == pair or (token[0] >= pair[0] and token[1] <= pair[1]):
                token_cats.append(cats_to_ids[cat])
                token_pols.append(pols_to_ids[pol])
                found = True
                break
        if found == False:
            token_cats.append(0)
            token_pols.append(0)
    return real_tokens, tokens, token_cats, token_pols


def predictor(raw, model_path, model, idx, out_path):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = [[None, None, None, None, 0, None, '']]
    ids_to_cats = {1:'Food', 2:'Interior', 3:'Price', 4:'Service', 5:'Whole'}
    for i in range(raw.shape[0]):
        starts, ends, cats, _, text = raw.iloc[i,:]
        pairs = list(twt().span_tokenize(text))
        tokens, labels, _ = model.predict(text, load_model=model_path, device=DEVICE)
        assert len(pairs) == len(tokens)
        for t_id in range(len(tokens)):
            if labels[t_id] != 0:
                #line = {'review_id':idx[i], 'category':ids_to_cats[labels[t_id]],
                #        'span':pairs[t_id][1]-pairs[t_id][0],
                #        'span_start':pairs[t_id][0], 'span_end':pairs[t_id][1], 'sentiment':None}


                #так как берт отмечал отдельные токены а не н-граммы, надо соединить последовательности в н_граммы, где они есть
                if ((out[-1][4] + 1) == pairs[t_id][0]) and (out[-1][1] == ids_to_cats[labels[t_id]]):
                    #приклеиваем новый токен к старому
                    line_update = [out[-1][0], out[-1][1], out[-1][2]+pairs[t_id][1]-pairs[t_id][0],
                                   out[-1][3], pairs[t_id][1], None, out[-1][-1]+tokens[t_id]]
                    out.pop()
                    out.append(line_update)
                else:
                    line = [idx[i], ids_to_cats[labels[t_id]], pairs[t_id][1]-pairs[t_id][0],
                            pairs[t_id][0], pairs[t_id][1], None, tokens[t_id]]

                    out.append(line)
    out = ['\t'.join(list(map(lambda x: str(x), tt))) for tt in out]
    with open(out_path, 'w') as f:
        f.write('\n'.join(out[1:]))
