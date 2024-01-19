
import datasets as ds
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import re
import numpy as np
from torchtext.vocab import Vectors
import nltk
from sklearn.manifold import TSNE
from scipy import spatial
import random
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score
# Commented out IPython magic to ensure Python compatibility.
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer
# %matplotlib inline

import nltk
from torchtext.legacy import data
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords
def get_data(dataset, vocab):
    data = []
    lens=[]
    for sentence in dataset:
            tokens = [vocab[token] for token in sentence]
            data.append(torch.LongTensor(tokens))
            lens.append(len(tokens))
    return data,lens

def _clean(i):
        i=i.lower()
        i = re.sub("[`]","", i)
        i = re.sub("''+","", i)
        i = re.sub("[^A-Za-z0-9' ]","", i)
        i = re.sub(" +"," ", i)
        return i
def clean(data):
  d=[]
  label=[]
  STOPWORDS = stopwords.words('english')
  for i in data:
    words=[token for token in tokenizer(_clean(i['sentence'])) if len(token)>=1 and token not in STOPWORDS and token != " "]
    words=["<sos>"]+words
    words.append("<eos>")
    if len(words)>0:
      d.append(words)
      label.append(torch.tensor(round(i['label'])).view(1))
    else:
      print(i)
  return d,label
TEXT=data.Field(use_vocab=True, lower=True, batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long, batch_first=True, sequential=False)
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
data_=ds.load_dataset("sst")
data_train=data_['train']
data_val=data_['validation']
data_test=data_['test']
train_tokens,train_labels=clean(data_train)
val_tokens,val_labels=clean(data_val)
test_tokens,test_labels=clean(data_test)
TEXT.build_vocab(train_tokens,
                 vectors = "glove.6B.300d", 
                 unk_init = torch.Tensor.normal_)

data_train_sst,train_lens=get_data(train_tokens,TEXT.vocab)
data_val_sst,val_lens=get_data(val_tokens,TEXT.vocab)
data_test_sst,test_lens=get_data(test_tokens,TEXT.vocab)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(ELMo, self).__init__()
        # glove_vectors= GloVe()
        self.embedding_dim=embedding_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True,dropout=0.3,bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                             batch_first=True,dropout=0.3,bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim,vocab_size)
        self.gamma = nn.Parameter(torch.ones(1))
        self.s = nn.Parameter(torch.zeros(3))
    def forward(self,embeds):
        
        lstm_1, _ = self.lstm(embeds)
        lstm_2, _ = self.lstm2(lstm_1)
        # Combine the ELMo representations from each layer
        lstm_embed_1=lstm_1.view(lstm_1.shape[0],lstm_1.shape[1],1,lstm_1.shape[2])
        lstm_embed_2=lstm_2.view(lstm_2.shape[0],lstm_2.shape[1],1,lstm_2.shape[2])
        embeds=embeds.view(embeds.shape[0],embeds.shape[1],1,embeds.shape[2])
        elmo_representations = torch.cat([lstm_embed_1, lstm_embed_2,embeds], dim=2)
        elmo_weights = F.softmax(self.s, dim=0).repeat(len(embeds),1,1,1).view(len(embeds),1,-1,1)  # (1, num_layers*2, 1)
        weighted_sum = torch.sum( elmo_representations*elmo_weights, dim=2)  # (seq_len, batch_size, 2*hidden_dim)
        weighted_sum = self.gamma * weighted_sum
        out = F.softmax(self.linear(weighted_sum))
        return out,weighted_sum

class SST(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(SST, self).__init__()
        # glove_vectors= GloVe()
        self.embedding_dim=embedding_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True,dropout=0.2,bidirectional=True)
        
        self.hidden_2_sig=nn.Linear(2*hidden_dim,2)
        
        self.drop=nn.Dropout(0.2)
    def forward(self,embeds):
        lstm, _ = self.lstm(embeds)
        out = self.hidden_2_sig(lstm[:,0,:])
        out = F.log_softmax(out)
        return out

from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
EMBEDDING_DIM = 150
HIDDEN_DIM = 150
EPOCHS = 10
LEARNING_RATE=5e-4
NUMBER_OF_LAYERS=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_batch_size=16
test_batch_szie=8
dataloader_train = DataLoader(list(zip(pad_sequence(data_train_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),train_labels)), batch_size=train_batch_size)
dataloader_val = DataLoader(list(zip(pad_sequence(data_val_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_labels)), batch_size=test_batch_szie)
dataloader_test = DataLoader(list(zip(pad_sequence(data_test_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),test_labels)), batch_size=test_batch_szie)
saved=False
model=ELMo(len(TEXT.vocab),2*EMBEDDING_DIM,HIDDEN_DIM)
model=model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()
loss_function=loss_function.to(device)

def train():
  for epoch in range(EPOCHS):
          train_loss=0
          tags=[]
          y_true=[]
          for batch in tqdm(dataloader_train):
            model.zero_grad()
            vectors=TEXT.vocab.vectors[batch[0][:,:-1]]
            vectors=vectors.to(device)
            y_actual=batch[0][:,1:]
            y_actual=y_actual.to(device)
            y_pred,embeds = model(vectors)
            y_pred=y_pred.reshape(y_actual.shape[0]*y_actual.shape[1],-1)
            y_actual=y_actual.reshape(-1)
            loss = loss_function(y_pred,y_actual)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
          print("train_loss",epoch,train_loss)
          valid_loss=0
          best_valid_loss = float('inf')          
          with torch.no_grad():
              for batch in dataloader_val:
                  vectors=TEXT.vocab.vectors[batch[0][:,:-1]]
                  vectors=vectors.to(device)
                  y_actual=batch[0][:,1:]
                  y_actual=y_actual.to(device)
                  y_pred,embeds = model(vectors)
                  y_pred=y_pred.reshape(y_actual.shape[0]*y_actual.shape[1],-1)
                  y_actual=y_actual.reshape(-1)
                  loss = loss_function(y_pred,y_actual)
                  valid_loss+=loss.item()
          if valid_loss < best_valid_loss:
              best_valid_loss = valid_loss
              torch.save(model.state_dict(), 'elmo_loves_you.pt')
          print("validation_loss",valid_loss)

train()

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
EMBEDDING_DIM = 150
HIDDEN_DIM = 150
EPOCHS = 30
LEARNING_RATE=1e-5
NUMBER_OF_LAYERS=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batch_size=64
test_batch_szie=8

loss_function = nn.CrossEntropyLoss()
loss_function=loss_function.to(device)

dataloader_train = DataLoader(list(zip(pad_sequence(data_train_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),train_labels)), batch_size=train_batch_size)
dataloader_val = DataLoader(list(zip(pad_sequence(data_val_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_labels)), batch_size=test_batch_szie)
dataloader_test = DataLoader(list(zip(pad_sequence(data_test_sst, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),test_labels)), batch_size=test_batch_szie)
model_sst=SST(len(TEXT.vocab),2*EMBEDDING_DIM,HIDDEN_DIM,)
model_sst=model_sst.to(device)
model.load_state_dict(torch.load('elmo_loves_you.pt'))
optimizer = torch.optim.Adam(model_sst.parameters(), lr=LEARNING_RATE)
_train_loss=[]
_val_loss=[]
def train():
  for epoch in range(EPOCHS):
          train_loss=0
          y_true=[]
          y_preds=[]
          print("EPOCH",epoch)
          print()
          for batch in tqdm(dataloader_train):
            model.zero_grad()
            vectors=TEXT.vocab.vectors[batch[0][:,:-1]]
            vectors=vectors.to(device)
            y_actual=batch[1].view(-1)
            y_actual=y_actual.to(device)
            elmo_embeds=model(vectors)[1]
            y_pred = model_sst(elmo_embeds)
            loss = loss_function(y_pred,y_actual)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            indices = torch.max(y_pred, 1)[1]
            y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
            y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
          print("train_loss",train_loss)
          print(classification_report(y_true,y_preds))
          print("========================================================================================================")
          _train_loss.append(train_loss)
          valid_loss=0
          y_true=[]
          y_preds=[] 
          best_valid_loss = float('inf')          
          with torch.no_grad():
              for batch in dataloader_val:
                  vectors=TEXT.vocab.vectors[batch[0][:,:-1]]
                  vectors=vectors.to(device)
                  y_actual=batch[1].view(-1)
                  y_actual=y_actual.to(device)
                  elmo_embeds=model(vectors)[1]
                  y_pred = model_sst(elmo_embeds)
                  loss = loss_function(y_pred,y_actual)
                  indices = torch.max(y_pred, 1)[1]
                  y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
                  y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
                  valid_loss+=loss.item()
                  if valid_loss < best_valid_loss:
                      best_valid_loss = valid_loss
                      torch.save(model_sst.state_dict(), 'sentiment_analysis_1.pt')
          print("validation_loss",valid_loss)
          _val_loss.append(valid_loss)
          print(classification_report(y_true,y_preds))
          print("========================================================================================================")
  plt.plot(np.arange(1, EPOCHS + 1), _train_loss, label='Train Loss', color='green')
  plt.plot(np.arange(1, EPOCHS + 1), _val_loss, label='Test Loss', color='red')
  plt.xlabel('EPOCH->')
  plt.ylabel('LOSS->')
  plt.legend()
  plt.show()


train()

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,auc
def test():
    model_sst.load_state_dict(torch.load('sentiment_analysis_1.pt'))
    valid_loss=0
    y_true=[]
    y_preds=[] 
    best_valid_loss = float('inf') 
    model_sst.eval()
    with torch.no_grad():
        for batch in dataloader_test:
            vectors=TEXT.vocab.vectors[batch[0][:,:-1]]
            vectors=vectors.to(device)
            y_actual=batch[1].view(-1)
            y_actual=y_actual.to(device)
            elmo_embeds=model(vectors)[1]
            y_pred = model_sst(elmo_embeds)
            loss = loss_function(y_pred,y_actual)
            indices = torch.max(y_pred, 1)[1]
            y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
            y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
            valid_loss+=loss.item()
            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     torch.save(model_sst.state_dict(), 'sentiment_analysis_1.pt')
    print("test_loss",valid_loss)
    print(classification_report(y_true,y_preds))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_preds)).plot()
    fpr, tpr, _ = roc_curve(y_true, y_preds)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('--------------------------------------')
test()

list(np.array([1,2,3]))

