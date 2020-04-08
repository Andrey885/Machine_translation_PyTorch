import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import argparse
import datetime
import json
from tqdm import tqdm
from data import prepare_data
from models import build_model

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion, writer, epoch, clip=1):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(iterator) + i)


    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_config', default=False)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--pf_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    if args.save_config:
        with open('./checkpoints/' + args.save_config, 'wt') as f:
            json.dump(vars(args), f, indent=4)

    train_data, valid_data, test_data, src_lang, trg_lang = prepare_data(args, device)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=args.batch_size,
        device=device)
    trg_pad_idx = trg_lang.vocab.stoi[trg_lang.pad_token]

    model = build_model(args, src_lang, trg_lang, len(src_lang.vocab), len(trg_lang.vocab), device)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    writer = SummaryWriter(log_dir='./logs/_{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()))

    for epoch in range(args.n_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion, writer, epoch)
        val_loss = evaluate(model, valid_iterator, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoints/en_de{epoch}.pt')

        writer.add_scalar('Loss/val', val_loss, epoch*len(train_iterator))
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')


if __name__ == '__main__':
    main()
