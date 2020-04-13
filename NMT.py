import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import os

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


def prepare_sequence(seq):
    # idxs = [to_ix[w] for w in seq]
    idxs = [w for w in seq]
    return torch.tensor(idxs, dtype=torch.long).cuda()

''' data example...
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
'''

def loadData(dictionary, lines, uselines):
    new_data = []
    use = []

    # filteredLines = [i for j, i in enumerate(lines) if j not in skiplines]
    length = len(lines)
    
    # index = 0
    # for line in filteredLines:
    for count in uselines:
        line = lines[count]
        '''
        if index in skiplines:
            print("skipping...")
            index += 1
            continue
        '''

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
            # dictionary.append(word)
            # converted_line.append(dictionary.index(word))

            # skips.append(index)
            # index += 1
            continue

        print((count / float(length)) * 100.00)
        new_data.append((sentence, converted_line))
        print(new_data[-1])
        # index += 1
    return new_data, dictionary, use
    # print(new_data[-1])

 ### Prepare data
print("preparing data...")
validationSet = ["newstest2012","newstest2013","newstest2014","newstest2015"]

sampleSize = 1
rootDir = "./datasets/"
englishDictionary = open(rootDir + "./vocab.50K.en.txt", 'r').read().lower().split("\n")
germanDictionary = open(rootDir + "./vocab.50K.de.txt", 'r').read().lower().split("\n")
englishTraining = open(rootDir + "./train.en", 'r').read().lower().split("\n")[::sampleSize]
germanEmbedding = open(rootDir + "./train.de", 'r').read().lower().split("\n")[::sampleSize]

savedEngData = []
savedGerData = []

if not os.path.exists(rootDir + "data-en-" + str(sampleSize) + ".json"):
    uselines = range(0, len(englishTraining))
    trainingData, englishDictionaryNew, uselines = loadData(englishDictionary, englishTraining, uselines)
    with open(rootDir + "data-en-" + str(sampleSize) + ".json", "w") as file:
        json.dump([trainingData, englishDictionaryNew, uselines], file)
    print("english data prepared...")

else:
    with open(rootDir + "data-en-" + str(sampleSize) + ".json", "r") as file:
        savedEngData = json.load(file)
    trainingData, englishDictionaryNew, uselines = savedEngData[0], savedEngData[1], savedEngData[2]

if not os.path.exists(rootDir + "data-de-" + str(sampleSize) + ".json"):
    trainingLabels, germanDictionaryNew, uselines = loadData(germanDictionary, germanEmbedding, uselines)
    with open(rootDir + "data-de-" + str(sampleSize) + ".json", "w") as file:
        json.dump([trainingLabels, germanDictionaryNew, uselines], file)
    print("german data prepared...")

else:
    with open(rootDir + "data-de-" + str(sampleSize) + ".json", "r") as file:
        savedGerData = json.load(file)
    trainingLabels, germanDictionaryNew, uselines = savedGerData[0], savedGerData[1], savedGerData[2]

print("consolidating lists...")


if not os.path.exists(rootDir + "data-together-" + str(sampleSize) + ".json"):
    print("saving entire dataset...")
    with open(rootDir + "data-together-" + str(sampleSize) + ".json", "w") as file:
        json.dump([savedEngData, savedGerData], file)

print("loaded data...")

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64 # 6
HIDDEN_DIM = 64 # 6

print("preparing model...")

# model = NMT(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = NMT(EMBEDDING_DIM, HIDDEN_DIM, len(savedEngData[0]), len(savedEngData[1])).cuda()
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

print("starting...")
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    
    index = 0
    _loss = 0

    # Training
    for sentence in savedEngData[0]:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

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
        if index % 100 == 0:
            print("\t",index,")", str(loss.item()))

        loss.backward()

        optimizer.step()

        index += 1
    
    print("Train ", epoch, " ) loss: ", str(_loss/len(savedEngData[0])))

    # Validation
    '''
    _loss = 0
    for sentence in savedEngDataVal[0]:
        model.train(False)
        model.eval()

        sentence_in = prepare_sequence(sentence[1])
        targets = prepare_sequence(savedEngData[0][index][1])

        with torch.no_grad():
            tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        _loss += loss.item()

        # b = blue

    print("Validation ", epoch, ") loss: ", _loss/len(savedEngDataVal[0]), " BLEU: ", b)
    model.train(True)
    '''

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
