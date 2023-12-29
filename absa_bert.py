import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

class ABSADataset():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sentiment_vocab = {'negative': 0, 'positive': 1, 'neutral': 2, 'both': 3}

    def __getitem__(self, index: int) -> tuple[tuple[str], int]:
        text = self.data['text'].values[index]
        sentiment = self.data['sentiment'].values[index]
        masked_text = (text[:self.data['span_start'].values[index]] +
                       '[MASK]' +
                       text[self.data['span_end'].values[index]:])
        if sentiment != 'None':
            return masked_text, self.sentiment_vocab[sentiment]
        else:
            return masked_text, -1

    def __len__(self) -> int:
        return len(self.data)


class ABSABert(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, 4)
        self.softmax = nn.Softmax()

    def forward(self, x, attention_mask=None):
        output = self.bert(x, attention_mask=attention_mask)['last_hidden_state']
        x = output[:, 0]
        x = self.dropout(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class ABSAModel():
    def __init__(self):
        self.model = ABSABert(AutoModel.from_pretrained('bert-base-multilingual-cased'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.trained = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model, path: str) -> None:
        model.load_state_dict(torch.load(path), strict=False)

    def save_model(self, model, name: str) -> None:
        torch.save(model.state_dict(), name)

    def train(self, data: pd.DataFrame, n_epoch: int, batch_size: int=5, lr: float=1e-6, load_model: str | None=None) -> None:

        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
                self.trained = True
            else:
                print("lead_model not found")

        self.model = self.model.to(self.device)

        train_data, val_data = train_test_split(data, test_size=0.25)

        self.train_dataloader = DataLoader(ABSADataset(train_data), batch_size=batch_size)
        self.val_dataloader = DataLoader(ABSADataset(val_data), batch_size=batch_size)

        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = []
        self.val_loss = []

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.train_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=0,
                                                               num_training_steps=len(self.train_dataloader) * n_epoch)

        for epoch in tqdm(range(n_epoch)):
            current_train_loss = 0
            for text, y in tqdm(self.train_dataloader, leave=False):

                tokens = self.tokenizer.batch_encode_plus(text,
                                                          padding='max_length',
                                                          return_tensors='pt')
                logits = self.model(tokens['input_ids'].to(self.device),
                                    attention_mask=tokens['attention_mask'].to(self.device))

                loss = self.criterion(logits, y.to(self.device))
                current_train_loss += loss.item()

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                self.train_scheduler.step()

            self.train_loss.append(current_train_loss / len(self.train_dataloader))

            current_val_loss = 0
            for text, y in self.val_dataloader:

                tokens = self.tokenizer.batch_encode_plus(text,
                                                          padding='max_length',
                                                          return_tensors='pt')

                logits = self.model(tokens['input_ids'].to(self.device),
                                    attention_mask=tokens['attention_mask'].to(self.device))
                loss = self.criterion(logits, y.to(self.device))
                current_val_loss += loss.item()
            self.val_loss.append(current_val_loss / len(self.val_dataloader))
            self.save_model(self.model, f'/content/drive/MyDrive/nlp/model_epochs{n_epoch}_batch{batch_size}.pkl')
            self.trained = True

    def predict(self, dev: pd.DataFrame, file_name: str, load_model: str | None=None) -> None:
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

        self.reverse_sentiment_vocab = {0: 'negative', 1: 'positive', 2: 'neutral', 3: 'both'}

        self.model = self.model.to(self.device)
        self.test_dataloader = DataLoader(ABSADataset(dev), batch_size=3)

        self.predictions = []

        with torch.no_grad():
            for text, y in tqdm(self.test_dataloader):
                tokens = self.tokenizer.batch_encode_plus(text,
                                                        padding='max_length',
                                                        return_tensors='pt')
                logits = self.model(tokens['input_ids'].to(self.device),
                                    attention_mask=tokens['attention_mask'].to(self.device)).argmax(dim=1)

                self.predictions += list(logits.data.cpu().numpy())

            self.predictions = [self.reverse_sentiment_vocab[item] for item in self.predictions]
            dev = dev.assign(sentiment=self.predictions).drop(columns=['text'])
            dev.to_csv(file_name, sep='\t', header=False)