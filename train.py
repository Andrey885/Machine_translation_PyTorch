import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator
from torch.utils.tensorboard.writer import SummaryWriter
import os
import argparse
import datetime
import glog as log
from tqdm import tqdm
from data import prepare_data
from models import build_model
from eval import evaluate, calculate_bleu


def train(model, iterator, optimizer, criterion, writer, epoch, clip=1):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(iterator) + i)
    return epoch_loss / len(iterator)


def main():
    parser = argparse.ArgumentParser(description='Training German-English translator model')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256, help='Model architecture parameter')
    parser.add_argument('--n_heads', type=int, default=8, help='Model architecture parameter')
    parser.add_argument('--n_layers', type=int, default=3, help='Model architecture parameter')
    parser.add_argument('--pf_dim', type=int, default=512, help='Model architecture parameter')
    parser.add_argument('--dropout', type=float, default=0.1, help='Model architecture parameter')

    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=args.batch_size,
        device=device)
    trg_pad_idx = trg_lang.vocab.stoi[trg_lang.pad_token]

    model = build_model(args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    os.makedirs('./logs', exist_ok=True)
    writer = SummaryWriter(log_dir='./logs/_{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()))
    for epoch in range(args.n_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion, writer, epoch)
        val_loss = evaluate(model, valid_iterator, criterion)
        bleu_score = calculate_bleu(valid_data, src_lang, trg_lang, model, device, max_len=100)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './checkpoints/en_de_best.pt')
        torch.save(model.state_dict(), f'./checkpoints/en_de{epoch}.pt')

        writer.add_scalar('Loss/val', val_loss, epoch*len(train_iterator))
        writer.add_scalar('Bleu_score', bleu_score, epoch*len(train_iterator))

        log.info(f'Epoch: {epoch}')
        log.info(f'\tTrain Loss: {train_loss:.3f}')
        log.info(f'\t Val. Loss: {val_loss:.3f}')
        log.info(f'\t Val. Bleu: {bleu_score:.3f}')


if __name__ == '__main__':
    main()
