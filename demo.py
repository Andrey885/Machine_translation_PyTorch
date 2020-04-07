import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import numpy as np
import argparse
import random
import math
import time
from tqdm import tqdm
from models import Encoder, Decoder, Seq2Seq
from train import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def main():

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    SRC = Field(tokenize = tokenize_de, init_token = '<sos>', eos_token = '<eos>',
                lower = True, batch_first = True)

    TRG = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>',
                lower = True, batch_first = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load('tut6-model.pt'))
    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
         batch_size = BATCH_SIZE,
         device = device)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    example_idx = 8

    src = vars(test_data.examples[example_idx])['src']
    trg = vars(test_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device)
    print(f'predicted trg = {translation}')




if __name__ == '__main__':
    main()
