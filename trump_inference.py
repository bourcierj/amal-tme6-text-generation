"""Generate text from a text generator trained on Trump's speeches dataset."""

import argparse
import torch

from trump_data import *
from rnn import TextGenerator
from utils import *
from generation import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Generate text from a text generator trained on Trump's speeches"
                        "dataset.")
        parser.add_argument('--saved-path', default='./checkpoints/saved/checkpt_best.pt',
                            type=str)
        parser.add_argument('--beginning', default='', type=str)
        parser.add_argument('--num-tokens', default=400, type=int)
        parser.add_argument('--top-k', default=1, type=int)
        return parser.parse_args()

    torch.manual_seed(42)
    args = parse_args()

    vocab = TrumpVocabulary()

    # load model from saved checkpoint
    net = TextGenerator(vocab.SIZE, embedding_size=250, hidden_size=250, cell='lstm')
    checkpoint = CheckpointState(net, savepath=args.saved_path)
    checkpoint.load()
    net = net.to(device)

    # Generate text wit greedy and beam search
    greedy_gen = generate_tokens_greedy(net, args.beginning, args.num_tokens, vocab)
    print('=> Greedy generated text:\n')
    print(args.beginning + greedy_gen)
    beam_search_gens = generate_tokens_beam_search(net, args.beginning, args.num_tokens,
                                                   args.top_k, vocab)
    print('\n=> Beam search generated text(s):\n')
    p = 0.
    for i, gen in enumerate(beam_search_gens, 1):
        print(f'Text {i}: \n{args.beginning + gen}\n')
