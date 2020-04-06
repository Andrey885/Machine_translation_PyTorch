import torch
import re
import random
import fasttext
import fasttext.util
import unicodedata
import pickle
import os
from models import Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_fasttext_embeddings_downloaded(langs):
    for lang in langs:
        if not os.path.exists('./data/embeddings/cc.'+lang+'.300.bin'):
            fasttext.util.download_model(lang, if_exists='ignore')  # English
            shutil.move('cc.'+lang+'.300.bin', './data/embeddings/cc.'+lang+'.300.bin')
            os.remove('cc.'+lang+'.300.bin.gz')


def dump_pickle(object, path):
    with open(path, 'wb') as file:
        pickle.dump(object, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length \
            and len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(args, reverse=False):
    lang1 = args.langs[0]
    lang2 = args.langs[1]
    input_lang, output_lang, pairs = readLangs(lang1, lang2, args.embedding_dimension, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs, args.max_sentence_length)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    random.Random(4).shuffle(pairs)  # shuffle pairs with seed 4
    pairs_train = pairs[:int(args.train_test_split * len(pairs))]
    pairs_test = pairs[int(args.train_test_split * len(pairs)):]
    return input_lang, output_lang, pairs_train, pairs_test


def readLangs(lang1, lang2, embedding_dimension = 300, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s/%s-%s.txt' % (lang1, lang2, lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    input_lang = Lang(lang1, embedding_dimension)
    output_lang = Lang(lang2, embedding_dimension)

    return input_lang, output_lang, pairs


def embeddingFromSentence(lang, sentence):
    return [torch.from_numpy(lang.pretrained_embedding.get_word_vector(word)) for word in sentence.split(' ')]


def langIndexFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(lang.word2index['EOS'])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = embeddingFromSentence(input_lang, pair[0])
    target_tensor = langIndexFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
