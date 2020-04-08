import spacy
from models import Encoder, Decoder, Seq2Seq
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


def prepare_data(args, device):
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

    # sudo python3 -m spacy download en
    # sudo python3 -m spacy download de_core_news_sm
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

    src_lang = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>',
                     lower=True, batch_first=True)

    trg_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>',
                     lower=True, batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(src_lang, trg_lang))
    src_lang.build_vocab(train_data, min_freq=2)
    trg_lang.build_vocab(train_data, min_freq=2)

    return train_data, valid_data, test_data, src_lang, trg_lang
