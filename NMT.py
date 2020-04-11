import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import os

torch.manual_seed(1)

class NMT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
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


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

''' data example...
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
'''

def loadData(dictionary, lines, skiplines):
    new_data = []
    skips = []

    filteredLines = [i for j, i in enumerate(lines) if j not in skiplines]
    length = len(filteredLines)
    
    index = 0
    for line in filteredLines:
        '''
        if index in skiplines:
            print("skipping...")
            index += 1
            continue
        '''

        sentence = line.replace("\n",'').split(" ")
        converted_line = []
        
        try:
            for word in sentence:
                converted_line.append(dictionary.index(word))

        except ValueError:
            # dictionary.append(word)
            # converted_line.append(dictionary.index(word))

            skips.append(index)
            index += 1
            continue

        print((index / float(length)) * 100.00)
        new_data.append((sentence, converted_line))
        print(new_data[-1])
        index += 1
    return new_data, dictionary, skips
    # print(new_data[-1])

 ### Prepare data
print("preparing data...")
validationSet = ["newstest2012","newstest2013","newstest2014","newstest2015"]

rootDir = "./datasets/"
englishDictionary = open(rootDir + "./vocab.50K.en.txt", 'r').read().lower().split("\n")
germanDictionary = open(rootDir + "./vocab.50K.de.txt", 'r').read().lower().split("\n")
englishTraining = open(rootDir + "./train.en", 'r').read().lower().split("\n")
germanEmbedding = open(rootDir + "./train.de", 'r').read().lower().split("\n")

savedEngData = []
savedGerData = []

if not os.path.exists(rootDir + "data-en.json"):
    skiplines = []
    trainingData, englishDictionaryNew, skiplines = loadData(englishDictionary, englishTraining, skiplines)
    with open(rootDir + "data-en.json", "w") as file:
        json.dump([trainingData, englishDictionaryNew, skiplines], file)
    print("english data prepared...")

else:
    with open(rootDir + "data-en.json", "r") as file:
        savedEngData = json.load(file)

if not os.path.exists(rootDir + "data-de.json"):
    trainnigLabels, germanDictionaryNew, skiplines = loadData(germanDictionary, germanEmbedding, skiplines)
    with open(rootDir + "data-de.json", "w") as file:
        json.dump([trainingLabels, germanDictionaryNew, skiplines], file)
    print("german data prepared...")

else:
    with open(rootDir + "data-de.json", "r") as file:
        savedGerData = json.load(file)

print("loaded data...")
exit()

word_to_it = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = NMT(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
'''
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
'''

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

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
