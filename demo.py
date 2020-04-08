import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import argparse
import os
import datetime
import json
from tqdm import tqdm
from data import prepare_data
from models import build_model
from eval_bleu import translate_sentence

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_name', default='config.json')
    parser.add_argument('--model_name', default='en_de_final.pt')

    args = parser.parse_args()

    with open(os.path.join('checkpoints', args.model_config_name), 'rt') as f:
        model_args = argparse.Namespace()
        model_args.__dict__.update(json.load(f))
        model_args = parser.parse_args(namespace=model_args)

    print('Loading models...')
    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data(model_args, device)
    model = build_model(model_args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)
    model.load_state_dict(torch.load('./checkpoints/'+args.model_name, map_location = 'cpu'))
    model.eval()

    print('Evaluating 5 random sentence from test set:')
    for _ in range(5):
        random_element = vars(test_data.examples[np.random.randint(len(test_data))])
        input_sentence = random_element['src']
        print('Input sentence: \n', ' '.join(input_sentence))
        translation, _ = translate_sentence(input_sentence, src_lang, trg_lang, model, device)
        print('GT translation: \n', ' '.join(random_element['trg']))
        print('Model translation: \n', ' '.join(translation))

if __name__ == '__main__':
    main()
