import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from datetime import datetime

import json
import os
import sys

from bleu import *
chencherry = SmoothingFunction()

torch.manual_seed(1)

def log(s):
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

    f = open('logfile/log.txt', 'a')
    f.write(timestamp + ": " + s.replace("\n","\n" + timestamp + ":") + '\n')

    f.close()

class NMT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(NMT, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim) 

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out1, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1)
        

        tag_space = self.hidden2tag(lstm_out2.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def preprocess(rootDir, sampleSize, englishDictionary, englishTraining, germanDictionary, germanEmbedding):
    if not os.path.exists(rootDir + "data-en-" + str(sampleSize) + ".json"):
        uselines = range(0, len(englishTraining))
        trainingData, englishDictionaryNew, uselines = loadData(englishDictionary, englishTraining, uselines)
        with open(rootDir + "data-en-" + str(sampleSize) + ".json", "w") as file:
            json.dump([trainingData, englishDictionaryNew, uselines], file)
    else:
        with open(rootDir + "data-en-" + str(sampleSize) + ".json", "r") as file:
            savedEngData = json.load(file)
        trainingData, englishDictionaryNew, uselines = savedEngData[0], savedEngData[1], savedEngData[2]
    
    # print("\t[ OK ] english data prepared...")
    log("\t[ OK ] english data prepared...")

    if not os.path.exists(rootDir + "data-de-" + str(sampleSize) + ".json"):
        trainingLabels, germanDictionaryNew, uselines = loadData(germanDictionary, germanEmbedding, uselines)
        with open(rootDir + "data-de-" + str(sampleSize) + ".json", "w") as file:
            json.dump([trainingLabels, germanDictionaryNew, uselines], file)

    else:
        with open(rootDir + "data-de-" + str(sampleSize) + ".json", "r") as file:
            savedGerData = json.load(file)
        trainingLabels, germanDictionaryNew, uselines = savedGerData[0], savedGerData[1], savedGerData[2]
    # print("\t[ OK ] german data prepared...")
    log("\t[ OK ] german data prepared...")

    '''
    print("consolidating lists...")
    if not os.path.exists(rootDir + "data-together-" + str(sampleSize) + ".json"):
        print("saving entire dataset...")
        with open(rootDir + "data-together-" + str(sampleSize) + ".json", "w") as file:
            json.dump([savedEngData, savedGerData], file)
    '''

    # print("\n\t[ OK ] loaded data...")
    log("\n\t[ OK ] loaded data...\n")
    # print()

    return savedEngData, savedGerData 

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def prepare_sequence(seq):
    # print("sequence: ", seq)
    idxs = [w for w in seq]
    return torch.tensor(idxs, dtype=torch.long).cuda()

''' data example...
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
'''

def convert(dictionary, data):
    s = ""
    # m = torch.max(data, dim=0)

    # print(data.shape)
    # print(m.shape)
    for itm in range(0, data.shape[0]):
        # print(torch.max(data[itm,:], dim=0)[1].detach().item())
        s += str(dictionary[torch.max(data[itm,:], dim=0)[1].detach().item()]) + " "
    return s


def loadModel(model, path):
    # model.load_state_dict(torch.load('./model/classify_cifar10_49.pth'))
    model.load_state_dict(torch.load(path))
    return model

def saveModel(model, path):
    torch.save(model.state_dict(), path)


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
    # print(new_data[-1])

### Prepare data
# print("preparing training data...")
log("preparing training data...")

sampleSize = 1
rootDir = "./datasets/"
englishDictionary = open(rootDir + "./vocab.50K.en.txt", 'r').read().lower().split("\n")
germanDictionary = open(rootDir + "./vocab.50K.de.txt", 'r').read().lower().split("\n")
englishTraining = open(rootDir + "./train.en", 'r').read().lower().split("\n")[::sampleSize]
germanEmbedding = open(rootDir + "./train.de", 'r').read().lower().split("\n")[::sampleSize]

savedEngData = []
savedGerData = []
savedEngVal = []
savedGerVal = []

savedEngData, savedGerData = preprocess(rootDir, sampleSize, englishDictionary, englishTraining, germanDictionary, germanEmbedding)

# validation data
validationSet = ["newstest2012","newstest2013","newstest2014","newstest2015"]
englishValidation = []
germanValidation = []
# print("preparing validation data...")
log("preparing validation data...")

for obj in validationSet:
    englishValidation = englishValidation + open(rootDir + "./" + obj + ".en.txt", 'r').read().lower().split("\n")
    germanValidation = germanValidation + open(rootDir + "./" + obj + ".de.txt", 'r').read().lower().split("\n")

    # print("\t[ OK ] ", obj)
    log("\t[ OK ] " + str(obj))

savedEngVal = loadData(englishDictionary, englishValidation, range(0, len(englishValidation)))
savedGerVal = loadData(germanDictionary, germanValidation, range(0, len(englishValidation)))

# print("\n\t[ OK ] loaded validation data")
log("\n\t[ OK ] loaded validation data")

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 128 # 128 # 6
HIDDEN_DIM = 128 # 128 # 6

# print("\npreparing model...")
log("\npreparing model...")

model = NMT(EMBEDDING_DIM, HIDDEN_DIM, len(savedEngData[0]), len(savedEngData[1])).cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

log("loading model...")
# model = loadModel(model,"./model/english_to_german_b8.191409480913228e-05-e0.pth")


#### Divide data for epochs
num_epochs = int(300 / sampleSize)
epoch_data = list(split(savedEngData[0], num_epochs))
# print(len(epoch_data), "epochs")
log(str(len(epoch_data)) + " epochs...")

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
'''

best_bleu = 0
index = 0
# print("starting...")
log("starting...")

# log("(from epoch 1)")
start_epoch = 0
for epoch in range(start_epoch, num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    
    _loss = 0
    _bleu = 0
    model.train(True)
    predictions = []
    groundTruth = []

    # Training
    for sentence in epoch_data[epoch]:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # print("sen: ", len(sentence))
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        # print("inputs: ", sentence[1])
        # print("tags: ", savedGerData[0][index][1])

        sentence_in = prepare_sequence(sentence[1])
        targets = prepare_sequence(savedEngData[0][index][1])

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        # print("\n\ntagscore:",tag_scores)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        
        # b = sentence_bleu([savedGerData[0][index][0]], convert(germanDictionary, tag_scores).split(" "), smoothing_function=chencherry.method1)
        # _bleu += b

        predictions.append(convert(germanDictionary, tag_scores).split(" "))
        groundTruth.append([["<s>"] + savedGerData[0][index][0] + ["</s>"]])
        
        # print("\ntest...")
        # print(convert(germanDictionary, tag_scores).split(" "))
        # print([["<s>"] + savedGerData[0][index][0] + ["</s>"]])        

        # loss = .5 * loss_function(tag_scores, targets) * (2*(1 - b))**2
        loss = loss_function(tag_scores, targets)
        _loss += loss.item()

        '''
        if index % 100 == 0:
            print("\t",index,")", str(loss.item()))
        '''
        loss.backward()
        optimizer.step()

        index += 1
   
    # calculate bleu score for training
    train_bleu = corpus_bleu(groundTruth, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1, auto_reweigh=False,)
    # train_bleu = _bleu

    # Validation
    model.train(False)
    model.eval()
    _bleu = 0

    groundTruth = []
    predictions = []

    val_index = 0;
    for sentence in savedEngVal[0]:
        sentence_in = prepare_sequence(sentence[1])
        targets = prepare_sequence(savedGerVal[0][val_index][1])

        with torch.no_grad():
            tag_scores = model(sentence_in)

        predictions.append(convert(germanDictionary, tag_scores).split(" "))
        groundTruth.append([["<s>"] + savedGerVal[0][val_index][0] + ["</s>"]])

        # print("sentence: ", " ".join(sentence[0]))
        # print("ground truth: ", " ".join(savedGerVal[0][val_index][0]))
        # print("predicted: ", convert(germanDictionary, tag_scores))

        # _bleu += sentence_bleu([savedGerVal[0][val_index][0]], convert(germanDictionary, tag_scores).split(" "), smoothing_function=chencherry.method1)
        val_index += 1

    _bleu = corpus_bleu(groundTruth, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1, auto_reweigh=False,) 

    po = str(epoch) + " ) training (loss): " + str(_loss/len(epoch_data[epoch])) + "  (bleu): " + str(train_bleu) + " | validation (bleu): " +  str(_bleu)  
    # print(epoch, ") training (loss): ", str(_loss/len(epoch_data[epoch])), "  (bleu): ", str(train_bleu), " | validation (bleu): ", _bleu) # , " BLEU: ", b)
    log(po)

    if _bleu > best_bleu:
        saveModel(model, "./model/english_to_german_b" + str(_bleu) + "-e" + str(epoch) + ".pth")
        # print("saving model...")
        log("saving model...")
        best_bleu = _bleu


# See what the scores are after training
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
'''
