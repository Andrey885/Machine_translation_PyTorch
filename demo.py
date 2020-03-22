import torch
import argparse
from models import EncoderRNN, AttnDecoderRNN
import data
import train
import random
from termcolor import colored

MAX_LENGTH = 15
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, help='Simple phrase in French to translate',
                        default='c\'était beau d\'essayer la traduction automatique')  # DEBUG: retrain
    parser.add_argument('--evaluate_random', type=bool, help='Translate 5 random sentences from dataset',
                        default=True)  # DEBUG: only one arg
    args = parser.parse_args()
    if args.sentence:
        args.evaluate_random = False

    input_lang = data.load_pickle('./data/eng-fra/input_lang.pkl')
    output_lang = data.load_pickle('./data/eng-fra/output_lang.pkl')
    # input_lang, output_lang, pairs_train, pairs_test = prepareData('eng', 'fra', MAX_LENGTH,
    #                                                                    0.66, True)
    # data.dump_pickle(input_lang, './data/eng-fra/input_lang.pkl')
    # data.dump_pickle(output_lang, './data/eng-fra/output_lang.pkl')
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
    encoder.load_state_dict(torch.load('./checkpoints/encoder_eng_fra7.pth', map_location=device))
    decoder.load_state_dict(torch.load('./checkpoints/decoder_eng_fra7.pth', map_location=device))

    if args.evaluate_random:
        lines = open('data/eng-fra/eng-fra.txt', encoding='utf-8').readlines()
        random.shuffle(lines)
        pairs = []
        for line in lines:
            if len(pairs) == 5:
                break
            pair = [data.normalizeString(s) for s in reversed(line.split('\t'))]
            if data.filterPair(pair, max_length=MAX_LENGTH):
                pairs.append(pair)
        for pair in pairs:
            decoded_words, _ = train.evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            output_sentence = ' '.join(decoded_words)
            print(colored('Input: ', 'yellow'), pair[0], colored('GT translation: ', 'green'), pair[1],
                  colored('Model translation: ', 'green'), output_sentence.replace('<EOS>', ''))
    else:
        line = args.sentence
        if len(line.split(' ')) >= MAX_LENGTH:
            raise NotImplementedError('Can\'t translate sentences longer than %d words' % MAX_LENGTH)
        line = data.normalizeString(line)
        decoded_words, _ = train.evaluate(encoder, decoder, line, input_lang, output_lang)
        output_sentence = ' '.join(decoded_words)
        print(colored('Input: ', 'yellow'), line, colored('Translation: ', 'green'),
              output_sentence.replace('<EOS>', ''))


if __name__ == '__main__':
    main()
