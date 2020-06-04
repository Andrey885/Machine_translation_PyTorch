import torch
import numpy as np
import argparse
import os
import json
from termcolor import colored
from data import prepare_data
from models import build_model
from utils import translate_sentence


def main():
    parser = argparse.ArgumentParser(description='demonstration of machine translation algorithm')
    parser.add_argument('--model_config', default='./checkpoints/config.json',
                        help='train config for model_weights')
    parser.add_argument('--model_weights', default='./checkpoints/en_de_final.pt',
                        help='path for weights of the model')
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(args.model_config), 'rt') as f:
        model_args = argparse.Namespace()
        model_args.__dict__.update(json.load(f))
        model_args = parser.parse_args(namespace=model_args)

    print('Loading models...')
    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data()
    model = build_model(model_args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)
    model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
    model.eval()

    print('Evaluating 5 random sentence from test set:')
    for _ in range(5):
        random_element = vars(test_data.examples[np.random.randint(len(test_data))])
        input_sentence = random_element['src']
        print(colored('Input sentence: \n', 'yellow'), ' '.join(input_sentence))
        translation, _ = translate_sentence(input_sentence, src_lang, trg_lang, model, device)
        # cut off <eos> token
        translation = translation[:-1]
        print(colored('GT translation: \n', 'green'), ' '.join(random_element['trg']))
        print(colored('Model translation: \n', 'green'), ' '.join(translation))


if __name__ == '__main__':
    main()
