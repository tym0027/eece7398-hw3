import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import os

from bleu import *
chencherry = SmoothingFunction()

torch.manual_seed(1)

class NMT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(NMT, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
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
    
    print("\t[ OK ] english data prepared...")

    if not os.path.exists(rootDir + "data-de-" + str(sampleSize) + ".json"):
        trainingLabels, germanDictionaryNew, uselines = loadData(germanDictionary, germanEmbedding, uselines)
        with open(rootDir + "data-de-" + str(sampleSize) + ".json", "w") as file:
            json.dump([trainingLabels, germanDictionaryNew, uselines], file)

    else:
        with open(rootDir + "data-de-" + str(sampleSize) + ".json", "r") as file:
            savedGerData = json.load(file)
        trainingLabels, germanDictionaryNew, uselines = savedGerData[0], savedGerData[1], savedGerData[2]
    print("\t[ OK ] german data prepared...")

    '''
    print("consolidating lists...")
    if not os.path.exists(rootDir + "data-together-" + str(sampleSize) + ".json"):
        print("saving entire dataset...")
        with open(rootDir + "data-together-" + str(sampleSize) + ".json", "w") as file:
            json.dump([savedEngData, savedGerData], file)
    '''

    print("\n\t[ OK ] loaded data...")

    print()

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
    for itm in range(1, data.shape[0] - 1):
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
    # print(new_data[-1])

### Prepare data
print("preparing training data...")

sampleSize = 10
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
print("preparing validation data...")

for obj in validationSet:
    englishValidation = englishValidation + open(rootDir + "./" + obj + ".en.txt", 'r').read().lower().split("\n")
    germanValidation = germanValidation + open(rootDir + "./" + obj + ".de.txt", 'r').read().lower().split("\n")

    print("\t[ OK ] ", obj)

savedEngVal = loadData(englishDictionary, englishValidation, range(0, len(englishValidation)))
savedGerVal = loadData(germanDictionary, germanValidation, range(0, len(englishValidation)))

print("\n\t[ OK ] loaded validation data")

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64 # 6
HIDDEN_DIM = 64 # 6

print("\npreparing model...")

model = NMT(EMBEDDING_DIM, HIDDEN_DIM, len(savedEngData[0]), len(savedEngData[1])).cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


#### Divide data for epochs
num_epochs = 30
epoch_data = list(split(savedEngData[0], num_epochs))
print(len(epoch_data), "epochs")

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
'''

index = 0
print("starting...")
for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    
    _loss = 0
    model.train(True)

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
        
        loss = loss_function(tag_scores, targets)
        

        _loss += loss.item()
        '''
        if index % 100 == 0:
            print("\t",index,")", str(loss.item()))
        '''
        loss.backward()

        optimizer.step()

        index += 1
    
    # print("Train ", epoch, " ) loss: ", str(_loss/len(epoch_data[epoch])))

    # Validation
    # _loss = 0
    model.train(False)
    model.eval()
    _bleu = 0

    val_index = 0;
    for sentence in savedEngVal[0]:
        sentence_in = prepare_sequence(sentence[1])
        targets = prepare_sequence(savedGerVal[0][val_index][1])

        with torch.no_grad():
            tag_scores = model(sentence_in)

        # print("sentence: ", " ".join(sentence[0]))
        # print("ground truth: ", " ".join(savedGerVal[0][val_index][0]))
        # print("predicted: ", convert(germanDictionary, tag_scores))

        _bleu += sentence_bleu([savedGerVal[0][val_index][0]], convert(germanDictionary, tag_scores).split(" "), smoothing_function=chencherry.method1)
        val_index += 1

    print(epoch, ") training (loss): ", str(_loss/len(epoch_data[epoch])), "  | validation (bleu): ", _bleu/len(savedEngVal[0])) # , " BLEU: ", b)

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
