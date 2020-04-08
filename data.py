import spacy
from models import Encoder, Decoder, Seq2Seq
import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
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

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=args.batch_size,
        device=device)

    input_dim = len(src_lang.vocab)
    output_dim = len(trg_lang.vocab)

    enc = Encoder(input_dim, args.hidden_size, args.n_layers,
                  args.n_heads, args.pf_dim, args.dropout,
                  device)

    dec = Decoder(output_dim, args.hidden_size, args.n_layers,
                  args.n_heads, args.pf_dim, args.dropout,
                  device)

    src_pad_idx = src_lang.vocab.stoi[src_lang.pad_token]
    trg_pad_idx = trg_lang.vocab.stoi[trg_lang.pad_token]

    model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    return model, train_iterator, valid_iterator, test_iterator, trg_pad_idx
