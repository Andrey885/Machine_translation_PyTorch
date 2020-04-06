import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import fasttext
import fasttext.util


class Lang:
    def __init__(self, name, embedding_dimension, load_embedding):
        self.name = name
        if load_embedding:
            self.pretrained_embedding = fasttext.load_model('./data/embeddings/cc.'+name+'.300.bin')

        if embedding_dimension != 300:
            fasttext.util.reduce_model(self.pretrained_embedding, embedding_dimension)
        self.word2index = {'SOS': 0, 'EOS':1}
        self.index2word = {0: "SOS", 1: "EOS"}

        self.n_words = 2  # Count SOS and EOS
        self.embedding_dimension = embedding_dimension

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        Add word to Lang vocabluary.
        Warning: all word in current dataset exist in current pretrained embedding.
        It may not be true if you decide to train on another dataset.
        """
        if word not in self.word2index:
            self.word2index[word] = 1
            self.index2word[self.n_words] = word

            self.n_words += 1


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, device = 'cpu'):
        super(EncoderRNN, self).__init__()
        # inputs are supposed to be pre-embedded
        self.device = device
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = 1, batch_first=True)

    def forward(self, embedded_input, hidden):
        embedded_input = embedded_input.unsqueeze(0)
        embedded_input = embedded_input.unsqueeze(0)
        output, hidden = self.gru(embedded_input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device = 'cpu', dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, 15)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
