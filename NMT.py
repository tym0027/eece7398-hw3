#### https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import sys

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)



def loadModel(model, path):
    # model.load_state_dict(torch.load('./model/classify_cifar10_49.pth'))
    model.load_state_dict(torch.load(path))
    return model


def saveModel(model, path):
    torch.save(model.state_dict(), path)


def convert_given(dictionary, data):
    s = ""
    for itm in range(0, data.shape[0]):
        s += str(dictionary[data[itm].item()]) + " "
    return s[:-1]

def convert(dictionary, data):
    s = ""
    # m = torch.max(data, dim=0)

    # print(data.shape)
    # print(m.shape)
    for itm in range(0, data.shape[0]):
        # print(torch.max(data[itm,:], dim=0)[1].detach().item())
        s += str(dictionary[torch.max(data[itm,:], dim=0)[1].detach().item()]) + " "
    return s[:-1]


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
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


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

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


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH #and \
        # p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    train_lines = open('data/%s-%s-training.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')[::10]
    valid_lines = open('data/%s-%s-validation.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    train_pairs = [[normalizeString(s) for s in l.split('\t')] for l in train_lines]
    valid_pairs = [[normalizeString(s) for s in l.split('\t')] for l in valid_lines]

    # Reverse pairs, make Lang instances
    if reverse:
        train_pairs = [list(reversed(p)) for p in train_pairs]
        valid_pairs = [list(reversed(p)) for p in valid_pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, train_pairs, valid_pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, train_pairs, valid_pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs (training)" % len(train_pairs))
    print("Read %s sentence pairs (validation)" % len(valid_pairs))
    train_pairs = filterPairs(train_pairs)
    valid_pairs = filterPairs(valid_pairs)
    print("Trimmed to %s sentence pairs (training)" % len(train_pairs))
    print("Trimmed to %s sentence pairs (validation)" % len(valid_pairs))
    print("Counting words...")
    for pair in train_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    for pair in valid_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, train_pairs, valid_pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, lang1, lang2, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    original = target_tensor 
    translated = torch.zeros((target_length, len(lang2.index2word.keys())))
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # ten.append(decoder_output)
            translated[di,:] = decoder_output

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # ten.append(decoder_output)
            translated[di,:] = decoder_output

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # print(translated.shape)
    # print(translated)

    # print(original.shape)
    # print(original)

    # print(decoder_output.shape)
    # print(convert(lang2.index2word, translated))
    # print(convert_given(lang2.index2word, original))
    
    hypothesis = convert(lang2.index2word, translated)
    reference = [convert_given(lang2.index2word, original)]
    # print("BLEU: ", sentence_bleu(reference, hypothesis, smoothing_function=chencherry.method1))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, hypothesis, reference, encoder, decoder


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def validation(encoder, decoder, lang1, lang2):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    validation_pairs = [tensorsFromPair(random.choice(valid_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    n_iters = len(valid_pairs)
    bleu = 0
    for iter in range(1, n_iters + 1):
        valid_pair = validation_pairs[iter - 1]
        input_tensor = valid_pair[0]
        target_tensor = valid_pair[1]

        loss, hypothesis, reference, encoder, decoder = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, lang1, lang2)
        print_loss_total += loss
        plot_loss_total += loss

        bleu += sentence_bleu(reference, hypothesis, smoothing_function=chencherry.method1)

    print('(VALID) %s (%d %d%%) loss: %.4f  bleu: %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, bleu / print_every))

    # showPlot(plot_losses)


    # for index in pairs:
    #    evaluate(encoder, decoder, pairs[index])

def trainIters(encoder, decoder, n_iters, lang1, lang2, print_every=1000, plot_every=100, learning_rate=0.01): # original learning_rate .01
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(train_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    bleu = 0
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, hypothesis, reference, encoder, decoder = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, lang1, lang2)
        print_loss_total += loss
        plot_loss_total += loss

        bleu += sentence_bleu(reference, hypothesis, smoothing_function=chencherry.method1)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(TRAIN) %s (%d %d%%) loss: %.4f  bleu: %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, bleu / print_every))

            saveModel(encoder, "./model/encoder-" + str(iter) + "-" + str(bleu/iter) + ".pth")
            saveModel(decoder, "./model/decoder-"+ str(iter) + "-" + str(bleu/iter) + ".pth")

            bleu = 0

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateSet(encoder, decoder, setlist):
    mean_bleu = 0
    bleu = 0
    for i in range(len(setlist)):
        pair = setlist[i]
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        # bleu = sentence_bleu(reference, hypothesis, smoothing_function=chencherry.method1)
        # mean_bleu += bleu

        print('<', output_sentence)
        print('~ bleu: ', bleu)
    print('\n\n~ mean bleu: ', mean_bleu/len(setlist))



def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


input_lang, output_lang, train_pairs, valid_pairs = prepareData('eng', 'ger', False)
print(random.choice(train_pairs))

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    trainIters(encoder1, attn_decoder1, 75000, input_lang, output_lang, print_every=5000)
elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    encoder1 = loadModel(encoder1, "./model/encoder-best.pth")
    attn_decoder1 = loadModel(attn_decoder1, "./model/decoder-best.pth")
    # validation(encoder1, attn_decoder1, input_lang, output_lang)
    evaluateSet(encoder1, attn_decoder1, valid_pairs)
elif len(sys.argv) > 1 and sys.argv[1] == 'translate':
    translation(encoder1, attn_decoder1, input_lang, output_lang)
