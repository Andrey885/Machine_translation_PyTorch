import torch
from torchtext.data.metrics import bleu_score
import argparse
import os
import json
from tqdm import tqdm
import glog as log
from data import prepare_data
from models import build_model
from utils import translate_sentence


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def calculate_bleu(data, src_field, trg_field, model, device, max_len=100):
    trgs = []
    pred_trgs = []
    for datum in tqdm(data):
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model_config', default='config.json',
                        help='train config for model_weights')
    parser.add_argument('--model_weights', default='./checkpoints/en_de_final.pt',
                        help='path for weights of the model')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join('checkpoints', args.model_config), 'rt') as f:
        model_args = argparse.Namespace()
        model_args.__dict__.update(json.load(f))
        model_args = parser.parse_args(namespace=model_args)

    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data()
    model = build_model(model_args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)
    model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
    model.eval()

    log.info('Bleu score: \n', calculate_bleu(test_data, src_lang, trg_lang, model, device, max_len=100))


if __name__ == '__main__':
    main()
