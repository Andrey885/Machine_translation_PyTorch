import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from data import prepare_data

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion, clip=1):
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
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs to train model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--pf_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    model, train_iterator, valid_iterator, test_iterator, trg_pad_idx = prepare_data(args, device)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    for epoch in range(args.n_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion)
        val_loss = evaluate(model, valid_iterator, criterion)

        if val_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), './checkpoints/en_de.pt')

        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()
