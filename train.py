import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
import os
import argparse
import datetime
import random
from tqdm import tqdm
import warnings
from models import EncoderRNN, DecoderRNN
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length):
    with torch.no_grad():
        input_tensor = data.embeddingFromSentence(input_lang, sentence)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(len(input_tensor)):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor(output_lang.word2index['SOS']).to(device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == output_lang.word2index['EOS']:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def eval(test_pairs, encoder, decoder, criterion, max_length, lang):
    test_loss = 0
    with torch.no_grad():
        for iteration in tqdm(range(len(test_pairs))):
            test_pair = test_pairs[iteration]
            input_tensor = test_pair[0]
            target_tensor = test_pair[1]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            loss = 0
            for ei in range(len(input_tensor)):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            decoder_input = torch.tensor(lang.word2index['SOS']).to(device)
            decoder_hidden = encoder_hidden
            # Without teacher forcing: use its own predictions as the next input
            for di in range(len(target_tensor)):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1, dim = 1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == lang.word2index['EOS']:
                    break
            test_loss += loss
    return test_loss.cpu().numpy() / len(test_pairs)


def trainIters(encoder, decoder, pairs_train, pairs_test, input_lang, output_lang, args):
    writer = SummaryWriter(log_dir='./logs/_{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)
    train_pairs = [data.tensorsFromPair(pairs_train[i], input_lang, output_lang)
                   for i in range(len(pairs_train))]
    test_pairs = [data.tensorsFromPair(pairs_test[i], input_lang, output_lang)
                  for i in range(len(pairs_test))]

    criterion = nn.NLLLoss()
    for epoch in range(args.n_epochs):
        for iteration in tqdm(range(len(train_pairs))):
            train_pair = train_pairs[iteration]
            input_tensor = train_pair[0]
            target_tensor = train_pair[1]
            loss = train_sentence(input_tensor, target_tensor, encoder,
                                decoder, encoder_optimizer, decoder_optimizer,
                                criterion, args.max_sentence_length, output_lang)
            writer.add_scalar('Loss/train', loss, epoch * len(train_pairs) + iteration)
        print('Train loss: ', loss)
        test_loss = eval(test_pairs, encoder, decoder, criterion, args.max_sentence_length, output_lang)
        print('Test loss: ', test_loss)
        pair = random.choice(pairs_test)
        decoded_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(decoded_words)
        print('Input: ', pair[0], 'GT translation: ', pair[1], 'Model translation: ', output_sentence)

        writer.add_scalar('Loss/test', test_loss, epoch)
        torch.save(encoder.state_dict(),
                   './checkpoints/encoder_' + args.langs[0] + '_' + args.langs[1] + repr(epoch) + '.pth')
        torch.save(decoder.state_dict(),
                   './checkpoints/decoder_' + args.langs[0] + '_' + args.langs[1] + repr(epoch) + '.pth')


def train_sentence(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                    max_length, lang, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(len(input_tensor)):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor(lang.word2index['SOS']).to(device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len(target_tensor)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(len(target_tensor)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1, dim = 1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == lang.word2index['EOS']:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / len(target_tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_drops', nargs='+', type=int, default=[100, 250, 350, 450], help='LR step sizes')
    parser.add_argument('--langs', nargs='+', type=str, default=['en', 'fr'],
                        help='Languages to train model (eng-fra supported)')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train model')
    parser.add_argument('--train_test_split', type=float, default=0.66, help='Ratio of train samples')
    parser.add_argument('--embedding_dimension', type=float, default=300, help='Word embedding dimension')
    parser.add_argument('--max_sentence_length', type=int, default=15, help = 'Max sentence length is used for attention mechanism')
    args = parser.parse_args()

    data.check_fasttext_embeddings_downloaded(args.langs)

    input_lang, output_lang, pairs_train, pairs_test = data.prepareData(args, True)

    encoder = EncoderRNN(args.embedding_dimension).to(device)
    decoder = DecoderRNN(args.embedding_dimension, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder, decoder, pairs_train, pairs_test, input_lang, output_lang, args)


if __name__ == '__main__':
    main()
