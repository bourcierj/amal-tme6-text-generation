"""Definition of PyTorch recurrent modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Long short term memory (LSTM) unit"""
    def __init__(self, input_size, hidden_size):

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # forget gate linear mapping
        self.lin_f = nn.Linear(input_size+hidden_size, hidden_size)
        # what and where to add to memory
        # entry gate linear mapping
        self.lin_i = nn.Linear(input_size+hidden_size, hidden_size)
        # entry with tanh
        self.lin_c = nn.Linear(input_size+hidden_size, hidden_size)
        # output gate linear mapping
        self.lin_o = nn.Linear(input_size+hidden_size, hidden_size)

    def one_step(self, xt, h, mem):
        # concatenates xt and h
        h_xt = torch.cat((h, xt), dim=1)
        ft = torch.sigmoid(self.lin_f(h_xt))
        it = torch.sigmoid(self.lin_i(h_xt))
        # memory state update
        out_mem = ft * mem + it * torch.tanh(self.lin_c(h_xt))
        # output gate
        ot = torch.sigmoid(self.lin_o(h_xt))
        out_h = ot * torch.tanh(out_mem)
        return out_h, out_mem

    def forward(self, x, state=None):
        if state is None:  # the state at previous time step
            # hidden state and memory state
            h = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
            mem = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        else:
            h, mem = state
            # the input hidden state must have size (1, batch, hidden)
            assert(h.size() == (1, x.size(1), self.hidden_size))
            # if (h.size(0) != x.size(1)):
            #     raise Exception("Provided hidden state (h) dimension 0 should"
            #                     f"match input dimension 1: got {h.size(0)} for h "
            #                     f"and {x.size(1)} for input")
            # if (mem.size(0) != x.size(1)):
            #     raise Exception("Provided memory state (c) dimension 0 should"
            #                     f"match input dimension 1: got {mem.size(0)} for c "
            #                     f"and {x.size(1)} for input")
        h.squeeze_(0)
        mem.squeeze_(0)
        output = list()
        for t in range(0, x.size(0)):
            h, mem = self.one_step(x[t, :, :], h, mem)
            output.append(h) # list of (1, batch, hidden) -> (T, batch, hidden)

        output = torch.stack(output, dim=0)
        h.unsqueeze_(0)
        mem.unsqueeze_(0)
        #output: (seq_length,batch,hidden), h and mem: (batch,hidden)
        return output, (h, mem)


class GRU(nn.Module):
    """Gated recurrent unit (GRU)"""
    def __init__(self, input_size, hidden_size):

        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lin_z = nn.Linear(input_size+hidden_size, hidden_size)
        self.lin_r = nn.Linear(input_size+hidden_size, hidden_size)
        self.lin_h = nn.Linear(input_size+hidden_size, hidden_size)

    def one_step(self, xt, h):
        # concatenates xt and h
        h_xt = torch.cat((h, xt), dim=1)
        zt = torch.sigmoid(self.lin_z(h_xt))
        rt = torch.sigmoid(self.lin_r(h_xt))
        # external state update
        out_h = (1 - zt) * h + zt * torch.tanh(self.lin_h(torch.cat((rt * h, xt), dim=1)))
        return out_h

    def forward(self, x, state=None):
        if state is None:  # the state at previous time step
            h = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)  # hidden state
        else:
            h, = state
            # the input hidden state must have size (1, batch, hidden)
            assert(h.size() == (1, x.size(1), self.hidden_size))
        h.squeeze_(0)
        output = list()
        for t in range(0, x.size(0)):
            h = self.one_step(x[t, :, :], h)
            output.append(h) # list of (1, batch, hidden) -> (T, batch, hidden)

        output = torch.stack(output, dim=0)
        h.unsqueeze_(0)
        return output, (h,)


class GRUWrapper(nn.GRU):
    """Wrapper around torch.nn.GRU.
    Overrides the forward pass to tak and output hidden states as tuples of one element.
    This unifies with torch.nn.LSTM inputs and outputs.
    """
    def __init__(self, *args, **kwargs):
        super(GRUWrapper, self).__init__(*args, **kwargs)

    def forward(self, input, state=None):
        if state is not None:
            h_0, = state
        else:
            h_0 = None
        out, h_n = super(GRUWrapper, self).forward(input, h_0)
        return out, (h_n,)


class TextGenerator(nn.Module):
    """Text generator using a recurrent cell (LSTM or GRU)
    Args:
        vocab_size (int): the size of the vocabulary, ie the size of the input to the
            embedding
        embedding_size (int): the size of the embedding layer
        hidden_size (int): the size of the hidden state of the RNN
        cell (str): the type of RNN cell to use. Can be either 'lstm' for an LSTM cell
            or 'gru' for a GRU cell.
        """
    def __init__(self, vocab_size, embedding_size, hidden_size, cell='lstm'):

        super(TextGenerator, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if cell == 'lstm':  # use an LSTM cell
            self.rnn = nn.LSTM(embedding_size, hidden_size)
        else:  # use a GRU cell
            self.rnn = GRUWrapper(embedding_size, hidden_size)
        # the output layer
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):

        embed = self.embedding(x)
        out, state = self.rnn(embed, state)
        out = self.lin(out)
        return out, state

    # def zero_state(self, batch_size):
    #     return (torch.zeros(1, batch_size, self.hidden_size),
    #             torch.zeros(1, batch_size, self.hidden_size))

if __name__ == '__main__':

    from pathlib import Path
    from trump_data import TrumpDataset, TrumpVocabulary
    from generation import *

    from torch.utils.data import DataLoader

    vocab = TrumpVocabulary()
    datapath = Path('../tme4-rnn/data/trump_full_speech.txt')
    dataset = TrumpDataset(datapath)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate)

    data, target = next(iter(loader))
    print(f"Input batch: {tuple(data.size())}")
    print(data)
    print(f"Target batch: {tuple(target.size())}")
    net = TextGenerator(vocab.SIZE, embedding_size=10, hidden_size=5, cell='lstm')
    output, _ = net(data)
    print(f"Output dim: {tuple(output.size())}")
    #print(output, '\n')

    gens = generate_tokens_greedy(net, '', 80, vocab)
    print(f"Generated text (greedy): {gens}")

    gens = generate_tokens_beam_search(net, '', 80, 5, vocab)
    print(f"Generated text (beam search): {gens}")

    # from process_trump import letter2id
    # print(letter2id)
    #I_code = letter2id['I']
    #gen = generator.generate_greedy(I_code, EOS_CODE)
    #print(code2string(gen, include_eos=True))
    #print(output.size())
    #print(output)

    # x : (seq_length, batch_size, input_size)
    # x = torch.randn(24, 16, 20)
    # lstm = LSTM(20, 8)
    # preds = lstm(x)
    # print(preds.size())
    # x = torch.randn(24, 16, 20)
    # gru = GRU(20, 8)
    # preds = gru(x)
    # print(preds.size())

