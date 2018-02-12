"""
This python file is try to use labelled 'Mode of Conviction' to build a RNN-based clasiffer for the mode of conviction

Author: Yuwei, Tu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


"""
Load Sample Data
"""

def _read_txtfile(txtpath, txt_filename):
    filename = txtpath + txt_filename
    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        #data = f.readlines()
        f = open(filename)
        data = f.readlines()
        f.close()
    return data

def load_data(sample_data):
    data = []
    for i in sample_data.index:
        example = {}
        txt_filename = sample_data.loc[i,'File']
        rawdata = _read_txtfile(txtpath, txt_filename)
        txtdata = ' '.join(rawdata.copy())
        example['text'] = txtdata.replace('\n', '')
        example['label'] = easy_label_map[sample_data.loc[i,'ModeOfConviction']]
        data.append(example)
    random.seed(1)
    random.shuffle(data)
    return data

sample = pd.read_csv('NY-Appellate-Scraping/2017-09-24/Archive/parsing2017/0.csv', sep=',',delimiter = ',', encoding='latin-1')
data = sample[sample['ModeOfConviction'].notnull()].reset_index()
train, test = train_test_split(data, test_size=0.2)
train, dev = train_test_split(train, test_size=0.2)

txtpath  = 'NY-Appellate-Scraping/2017-09-10/courtdoc/txt/'
easy_label_map = {"plea of guilty":0, "jury verdict":1, "nonjury trial":2}

training_set = load_data(train)
dev_set = load_data(dev)
test_set = load_data(test)

print('Number of training samples: ', len(training_set))
print('Number of dev samples: ', len(dev_set))
print('Number of test samples: ', len(test_set))


"""
Pad and Index
"""

import collections
import numpy as np

PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 20


def tokenize(string):
    return string.split()


def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))

    vocabulary = set([word for word in word_counter if word_counter[word] > 10])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)


def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1, -1)
            example['label'] = torch.LongTensor([example['label']])



word_to_ix, vocab_size = build_dictionary([training_set])
sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set])

"""
Batchify Data
"""


# This is the iterator we'll use during training.
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]


# This is the iterator we use when we're evaluating our model.
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue

    return batches


# The following function gives batches of vectors and labels,
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels

"""
Model
"""


class ElmanRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, batch_size):
        super(ElmanRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def forward(self, x, hidden):
        x_emb = self.embed(x)
        embs = torch.chunk(x_emb, x_emb.size()[1], 1)

        def step(emb, hid):
            combined = torch.cat((hid, emb), 1)
            hid = F.tanh(self.i2h(combined))
            return hid

        for i in range(len(embs)):
            hidden = step(embs[i].squeeze(), hidden)

        output = self.decoder(hidden)
        return output, hidden

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        return h0

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.i2h, self.decoder]
        em_layer = [self.embed]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


"""
Training
"""
# Hyper Parameters
input_size = vocab_size
num_labels = 4
hidden_dim = 24
embedding_dim = 8
batch_size = 25
learning_rate = 0.2
num_epochs = 200


# Build, initialize, and train model
rnn = ElmanRNN(vocab_size, embedding_dim, hidden_dim, num_labels, batch_size)
rnn.init_weights()

# Loss and Optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# Train the model
training_iter = data_iter(training_set, batch_size)
#train_eval_iter = eval_iter(training_set[0:500], batch_size)
dev_iter = eval_iter(dev_set, batch_size)


"""
To Be Finished
"""
