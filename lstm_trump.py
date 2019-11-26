"""Trains an LSTM or GRU text generator on the Trump speeches dataset."""

from pathlib import Path
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from trump_data import *
from rnn import TextGenerator
from utils import *
from generation import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(checkpoint, criterion, loader, epochs, clip=None,
          writer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    if clip is None:  # no gradient clipping
        clip = float('inf')
    min_loss = float('inf')
    iteration = 1
    LOG_TEXTGEN = 1  # log generated text every X epochs
    vocab = TrumpVocabulary()

    def train_epoch():
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True)  # progress bar
        net.train()
        for data, target in pbar:

            data, target = data.to(device), target.to(device)
            seq_length, batch_size = tuple(data.size())
            # reset gradients
            optimizer.zero_grad()
            output, _ = net(data) # (seq_length, batch, vocab_size)

            # flatten output and target to compute loss on individual words
            loss = criterion(output.view(batch_size*seq_length, -1), target.view(-1))
            # loss = criterion(output.transpose(1, 2), target)
            # print(f"Loss: {loss:.4f}")
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Loss/Train-batch-loss', loss.item(), iteration)
            # compute gradients, update parameters
            loss.backward()
            # Gradient clipping helps prevent the exploding gradient problem in RNNs
            # clip the gradients to the given clip value (+inf if not specified),
            # return the total norm of parameters
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                        max_norm=clip)
            if writer:
                writer.add_scalar("Total-norm-of-parameters", total_norm, iteration)

            optimizer.step()
            iteration += 1

        epoch_loss /= len(loader)
        print(f'Epoch {epoch}/{epochs}, Mean loss: {epoch_loss:.4e}')
        if writer:
            writer.add_scalar('Loss/Train-epoch-loss', epoch_loss, epoch)
        return epoch_loss

    begin_epoch = checkpoint.epoch
    for epoch in range(begin_epoch, epochs+1):

        loss = train_epoch()
        checkpoint.epoch += 1
        if loss < min_loss:
            min_loss = loss
            min_epoch = epoch
            checkpoint.save('_best')
        checkpoint.save()

        if writer and epoch % LOG_TEXTGEN == 0:
            # Generate text and log it to tensorflow (with greedy and beam search)
            greedy_gen = generate_tokens_greedy(net, 'I am ', 220, vocab)
            beam_search_gens = generate_tokens_beam_search(net, 'I am', 220, 3, vocab)

            writer.add_text('Generated/Greedy', 'I am '+greedy_gen, epoch)
            for i, gen in enumerate(beam_search_gens, 1):
                writer.add_text(f'Generated/Beam-search-{i}', gen, epoch)

    print("\nFinished.")
    print(f"Best loss: {min_loss:.4e}\n")
    print(f"Best epoch: {min_epoch}")
    return


if __name__ == '__main__':

    def parse_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Trains an LSTM RNN text generator on the Trump speeches dataset.")
        parser.add_argument('--no-tensorboard', action='store_true')
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--clip', default=None, type=float)
        parser.add_argument('--cell', default='lstm', type=str, choices=['gru', 'lstm'])
        parser.add_argument('--embedding-size', default=30, type=int)
        parser.add_argument('--hidden-size', default=10, type=int)
        return parser.parse_args()

    torch.manual_seed(42)
    args = parse_args()

    vocab = TrumpVocabulary()
    loader = get_dataloader(args.batch_size)

    net = TextGenerator(vocab.SIZE, args.embedding_size, args.hidden_size, args.cell)
    net = net.to(device)

    # In order to exclude losses computed on null entries (zero),
    # set ignore_index=0 for the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ignore_keys = {'no_tensorboard'}
    # get hyperparameters with values in a dict
    hparams = {key.replace('_','-'): val for key, val in vars(args).items()
               if key not in ignore_keys}
    # generate a name for the experiment
    expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
    # path where to save the model
    savepath = Path('./checkpoints/checkpt.pt')
    # Tensorboard summary writer
    if args.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(comment='__Trump-LSTM__'+expe_name, flush_secs=10)

    checkpoint = CheckpointState(net, optimizer, savepath=savepath)

    train(checkpoint, criterion, loader, args.epochs, clip=args.clip, writer=writer)

    if writer:
        writer.close()
