import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import datetime
import random
from tqdm import tqdm
from models import EncoderRNN, AttnDecoderRNN
from data import tensorFromSentence, prepareData, tensorsFromPair

MAX_LENGTH = 15
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, EOS_token)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def trainIters(encoder, decoder, pairs_train, pairs_test, input_lang, output_lang, args):
    writer = SummaryWriter(log_dir='./logs/_{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)
    train_pairs = [tensorsFromPair(pairs_train[i], input_lang, output_lang, EOS_token)
                   for i in range(len(pairs_train))]
    test_pairs = [tensorsFromPair(pairs_test[i], input_lang, output_lang, EOS_token)
                  for i in range(len(pairs_test))]
    criterion = nn.NLLLoss()
    for epoch in range(args.n_epochs):
        for iteration in tqdm(range(len(train_pairs))):
            train_pair = train_pairs[iteration]
            input_tensor = train_pair[0]
            target_tensor = train_pair[1]
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            writer.add_scalar('Loss/train', loss, epoch * len(train_pairs) + iteration)
        print('Train loss: ', loss)
        test_loss = eval(test_pairs, encoder, decoder, criterion)
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


def eval(test_pairs, encoder, decoder, criterion):
    test_loss = 0
    with torch.no_grad():
        for iteration in tqdm(range(len(test_pairs))):
            test_pair = test_pairs[iteration]
            input_tensor = test_pair[0]
            target_tensor = test_pair[1]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
            loss = 0
            for ei in range(input_tensor.shape[0]):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_tensor.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
            test_loss += loss
    return test_loss.cpu().numpy() / len(test_pairs)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_tensor.shape[0]):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_tensor.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_tensor.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_drops', nargs='+', type=int, default=[100, 250, 350, 450], help='LR step sizes')
    parser.add_argument('--langs', nargs='+', type=str, default=['eng', 'fra'],
                        help='Languages to train model (eng-fra supported)')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train model')
    parser.add_argument('--train_test_split', type=float, default=0.66, help='Ratio of train samples')
    args = parser.parse_args()
    input_lang, output_lang, pairs_train, pairs_test = prepareData(args.langs[0], args.langs[1], MAX_LENGTH,
                                                                   args.train_test_split, True)
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)

    trainIters(encoder, attn_decoder, pairs_train, pairs_test, input_lang, output_lang, args)


if __name__ == '__main__':
    main()
