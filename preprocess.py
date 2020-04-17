#### https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
from datetime import datetime
import math
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import os
import sys
import io
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bleu import *
chencherry = SmoothingFunction()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"]='1,3,6,7'

# Random Seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

def log(s):
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

    f = open('logfile/log.txt', 'a')
    f.write(timestamp + ": " + s.replace("\n","\n" + timestamp + ":") + '\n')

    f.close()

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # output, hidden = self.lstm(output, hidden.unsqueeze(0))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class NMT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, MAX_LENGTH):
        super(NMT, self).__init__()
        self.hidden_dim = hidden_dim

        self.encode = EncoderRNN(vocab_size, embedding_dim)

        self.decode = AttnDecoderRNN(hidden_dim, tagset_size, dropout_p=0.1, max_length=MAX_LENGTH)

    def forward(self, sentence, encoder_hidden):
        encoder_outputs, encoder_hidden = self.encode(sentence, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token]], device='cuda')
        tag_scores, hidden, attn_weights = self.decode(decoder_input, decoder_hidden, encoder_outputs)
        '''
        embeds = self.word_embeddings(sentence)
        lstm_out1, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1)
        tag_space = self.hidden2tag(lstm_out2.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        '''
        return tag_scores



def convert(dictionary, data):
    s = ""
    # m = torch.max(data, dim=0)

    # print(data.shape)
    # print(m.shape)
    for itm in range(0, data.shape[0]):
        # print(torch.max(data[itm,:], dim=0)[1].detach().item())
        s += str(dictionary[torch.max(data[itm,:], dim=0)[1].detach().item()]) + " "
    return s


def loadData(dictionary, lines, uselines):
    new_data = []
    use = []
    length = len(lines)

    for count in uselines:
        line = lines[count]

        sentence = line.replace("\n",'').split(" ")
        converted_line = []

        try:
            converted_line.append(1)
            for word in sentence:
                try:
                    converted_line.append(dictionary.index(word))
                except ValueError:
                    converted_line.append(0)
            converted_line.append(2)
            use.append(count)

        except:
            continue

        # print((count / float(length)) * 100.00)
        new_data.append((sentence, converted_line))
        # print(new_data[-1])
    return new_data, dictionary, use


def prepare_sequence(seq):
    # print("sequence: ", seq)
    idxs = [w for w in seq]
    return torch.tensor(idxs, dtype=torch.long).cuda()


def preprocess(rootDir, lang1, lang2, sampleSize, englishDictionary, englishTraining, targetDictionary, targetEmbedding):
    if not os.path.exists(rootDir + "data-" + lang1 + "-" + str(sampleSize) + ".json"):
        uselines = range(0, len(englishTraining))
        savedEngData = loadData(englishDictionary, englishTraining, uselines)
        with open(rootDir + "data-" + lang1 + "-" + str(sampleSize) + ".json", "w", encoding="utf-8") as file:
            json.dump(savedEngData, file)
        '''
        with io.open(rootDir + "data-" + lang1 + "-" + str(sampleSize) + ".json",'w',encoding="utf-8") as outfile:
            outfile.write(unicode(json.dumps([trainingData, englishDictionaryNew, uselines], ensure_ascii=False)))
        '''
    else:
        with open(rootDir + "data-" + lang1 + "-" + str(sampleSize) + ".json", "r", encoding="utf-8") as file:
            savedEngData = json.load(file)
        # trainingData, englishDictionaryNew, uselines = savedEngData[0], savedEngData[1], savedEngData[2]

    log("\t[ OK ] english data prepared...")

    if not os.path.exists(rootDir + "data-" + lang2 + "-"  + str(sampleSize) + ".json"):
        print("fuck me")
        savedTarData = loadData(targetDictionary, targetEmbedding, uselines)
        with open(rootDir + "data-" + lang2 + "-" + str(sampleSize) + ".json", "w", encoding="utf-8") as file:
            json.dump(savedTarData, file)

    else:
        with open(rootDir + "data-" + lang2 + "-" + str(sampleSize) + ".json", "r", encoding="utf-8") as file:
            savedTarData = json.load(file)
        # trainingLabels, targetDictionaryNew, uselines = savedTarData[0], savedTarData[1], savedTarData[2]
    log("\t[ OK ] target data prepared...")

    log("\n\t[ OK ] loaded data...\n")
    return savedEngData, savedTarData


### Prepare data
log("preparing training data...")

sampleSize = 10
rootDir = "./data/english-german/"
lang1 = "en"
lang2 = "de"
englishDictionary = open(rootDir + "./vocab.50K." + lang1 + ".txt", 'r').read().lower().split("\n")
targetDictionary = open(rootDir + "./vocab.50K." + lang2 + ".txt", 'r').read().lower().split("\n")
# englishDictionary = open(rootDir + "vocab." + lang1, 'r', encoding="utf-8").read().lower().split("\n")
# targetDictionary = open(rootDir + "vocab." + lang2, 'r', encoding="utf-8").read().lower().split("\n")
englishTraining = open(rootDir + "./train." + lang1, 'r', encoding="utf-8").read().lower().split("\n")[::sampleSize]
targetEmbedding = open(rootDir + "./train." + lang2, 'r', encoding="utf-8").read().lower().split("\n")[::sampleSize]

savedEngData = []
savedTarData = []
savedEngVal = []
savedTarVal = []

savedEngData, savedTarData = preprocess(rootDir, lang1, lang2, sampleSize, englishDictionary, englishTraining, targetDictionary, targetEmbedding)

# validation data
validationSet = ["newstest2012","newstest2013","newstest2014","newstest2015"]
# validationSet = ["tst2012", "tst2013"]
englishValidation = []
targetValidation = []
# print("preparing validation data...")
log("preparing validation data...")

for obj in validationSet:
    englishValidation = englishValidation + open(rootDir + "./" + obj + "." + lang1 + ".txt", 'r', encoding="utf-8").read().lower().split("\n")
    targetValidation = targetValidation + open(rootDir + "./" + obj + "." + lang2 + ".txt", 'r', encoding="utf-8").read().lower().split("\n")

    # print("\t[ OK ] ", obj)
    log("\t[ OK ] " + str(obj))

savedEngVal = loadData(englishDictionary, englishValidation, range(0, len(englishValidation)))
savedTarVal = loadData(targetDictionary, targetValidation, range(0, len(targetValidation)))

# print("\n\t[ OK ] loaded validation data")
log("\n\t[ OK ] loaded validation data")

# translate training data into seq2seq tutorial format...
save_train_file = open(rootDir + "/training.txt", 'w', encoding="utf-8")
for index in range(0, len(savedEngData[0])):
    s = ""
    for word in savedEngData[0][index][0]:
        s = s + str(word) + " "
    s = s + "\t"

    for word in savedTarData[0][index][0]:
        s = s + str(word) + " "
    # s = s + "\n"
    print(s)
    save_train_file.write(s + "\n")

save_train_file.close()

# translate validation data into seq2seq tutorial format
save_valid_file = open(rootDir + "/validation.txt", 'w', encoding="utf-8")
for index in range(0, len(savedEngVal[0])):
    s = ""
    for word in savedEngVal[0][index][0]:
        s = s + str(word) + " "
    s = s + "\t"

    for word in savedTarVal[0][index][0]:
        s = s + str(word) + " "
    # s = s + "\n"
    print(s)
    save_valid_file.write(s + "\n")

save_valid_file.close()


