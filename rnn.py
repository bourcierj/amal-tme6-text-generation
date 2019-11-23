"""Definition of PyTorch recurrent modules."""

import heapq

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

        if not state:  # the state at previous time step
            # hidden state and memory state
            h = torch.zeros(x.size(1), self.hidden_size).to(x.device)
            mem = torch.zeros(x.size(1), self.hidden_size).to(x.device)
        else:
            h, mem = state
            # if (h.size(0) != x.size(1)):
            #     raise Exception("Provided hidden state (h) dimension 0 should"
            #                     f"match input dimension 1: got {h.size(0)} for h "
            #                     f"and {x.size(1)} for input")
            # if (mem.size(0) != x.size(1)):
            #     raise Exception("Provided memory state (c) dimension 0 should"
            #                     f"match input dimension 1: got {mem.size(0)} for c "
            #                     f"and {x.size(1)} for input")
        output = list()
        for t, xt in enumerate(x):
            h, mem = self.one_step(xt, h, mem)
            output.append(h)

        output = torch.stack(output, dim=0)
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

        if not state:  # the state at previous time step
            h = torch.zeros(x.size(1), self.hidden_size).to(x.device)  # hidden state
        else:
            h, = state,
        output = list()
        for t, xt in enumerate(x):
            h = self.one_step(xt, h)
            output.append(h)

        output = torch.stack(output, dim=0)
        return output, (h,)


class TextGenerator(nn.Module):
    """Text generator using a recurrent cell (RNN, LSTM or GRU)"""
    def __init__(self, vocab_size, embedding_size, hidden_size):

        super(TextGenerator, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        if not state:
            h = torch.zeros(x.size(1), self.hidden_size).to(x.device)
            c = torch.zeros(x.size(1), self.hidden_size).to(x.device)
            state = (h, c)

        embed = self.embedding(x) # (T, B, embedding_size)
        out, _ = self.rnn(embed, state) # (T, B, hidden_size)
        out = self.lin(out) # (T, B, vocab_size)
        return out, state

    # def zero_state(self, batch_size):
    #     return (torch.zeros(1, batch_size, self.hidden_size),
    #             torch.zeros(1, batch_size, self.hidden_size))

if __name__ == '__main__':

    from pathlib import Path
    from trump_data import TrumpDataset, TrumpVocabulary

    from torch.utils.data import DataLoader

    vocab = TrumpVocabulary()
    datapath = Path('../tme4/data/trump_full_speech.txt')
    dataset = TrumpDataset(datapath)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate)

    data, target = next(iter(loader))
    print(f"Input batch: {tuple(data.size())}")
    print(data)
    print(f"Target batch: {tuple(target.size())}")
    net = TextGenerator(vocab.SIZE, embedding_size=10, hidden_size=5)
    output, _ = net(data)
    print(f"Output dim: {tuple(output.size())}")
    #print(output, '\n')

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

