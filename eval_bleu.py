import torch
from torchtext.data.metrics import bleu_score
import argparse
import os
import datetime
import json
from tqdm import tqdm
from data import prepare_data
from models import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
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


def calculate_bleu(data, src_field, trg_field, model, device, max_len = 100):
    trgs = []
    pred_trgs = []
    for datum in tqdm(data):
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config.json')
    args = parser.parse_args()

    with open(os.path.join('checkpoints', args.model_config), 'rt') as f:
        model_args = argparse.Namespace()
        model_args.__dict__.update(json.load(f))
        model_args = parser.parse_args(namespace=model_args)

    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data(model_args, device)
    model = build_model(model_args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)
    model.load_state_dict(torch.load('./checkpoints/en_de_final.pt', map_location = 'cpu'))
    model.eval()

    print('Bleu score: \n', calculate_bleu(test_data, src_lang, trg_lang, model, device, max_len = 100))
