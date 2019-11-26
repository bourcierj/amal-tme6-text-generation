import torch
import torch.nn.functional as F

def generate_sentences_greedy(net, beginning, num_sentences, vocab):
    net.eval()
    input = vocab.string2code(beginning).view(-1, 1)  # input
    input = input.to(next(net.parameters()).device)
    gens = list()
    num_sentences_gen = 0
    state = None
    with torch.no_grad():
        while num_sentences_gen < num_sentences:
            out, state = net(input, state)
            # If this is the prediction for the beginning tokens, keep only last
            # time step
            out = out[-1, :, :]  # note: the first dimension is removed
            # extract the predicted token
            token = out.argmax()
            gens.append(token.item())
            input = token.view(-1, 1)
            if token.item() == vocab.EOS_ID:
                num_sentences_gen += 1

    gens = vocab.code2string(gens)
    return gens


def generate_tokens_greedy(net, beginning, num_tokens, vocab):
    """Generates tokens by using a greedy search. Takes the argmax of the predicted
    distribution over the tokens.
    Args:
        net (nn.Module): the generator
        beginning (str): a string to begin the text generation. It can provide a
            context via a hidden state for the first token generated. Give a empty
            string to begin generation without context.
        num_tokens (int): the number of tokens to generate.
        vocab (TrumpVocabulary): the vocabulary

    Returns:
        str: the generated text
    """
    net.eval()
    input = vocab.string2code(beginning).view(-1, 1)  # input
    input = input.to(next(net.parameters()).device)
    gens = list()
    num_tokens_gen = 0
    state = None
    with torch.no_grad():
        while num_tokens_gen < num_tokens:
            out, state = net(input, state)
            # If this is the prediction for the beginning tokens, keep only last
            # time step
            out = out[-1, :, :]  # note: the first dimension is removed
            out = F.log_softmax(out, 1)
            # extract the predicted token
            token = out.argmax()
            gens.append(token.item())
            input = token.view(-1, 1)
            num_tokens_gen += 1

    text = vocab.code2string(gens)
    return text


def generate_tokens_beam_search(net, beginning, num_tokens, top_k, vocab):
    """Generates tokens by using a beam search.
    Args:
        net (nn.Module): the generator
        beginning (str): a string to begin the text generation. It can provide a
            context via a hidden state for the first token generated. Give a empty
            string to begin generation without context.
        num_tokens (int): the number of tokens to generate.
        top_k (int): the beam seach width: number of candidates to consider
        vocab (TrumpVocabulary): the vocabulary
    Returns:
        str: the generated text
    """
    class BeamSearchNode():
        def __init__(self, rnn_state, prev_node, token, logp):
            self.rnn_state = rnn_state
            self.prev_node = prev_node
            self.token = token
            self.logp = logp
            if prev_node is None:
                self.length = 1
            else:
                self.length = prev_node.length + 1

        def eval(self):
            return self.logp
            # reward = 0
            # return self.logp / float(self.length + 1 + 1e-8) + alpha * reward

        def __lt__(self, other):
            return self.eval() < other.eval()

    # @todo: manage when to stop generating tokens, after an EOS
    net.eval()
    input = vocab.string2code(beginning).view(-1, 1)  # input
    input = input.to(next(net.parameters()).device)
    gens = list()
    num_tokens_gen = 0
    state = None
    # beam search nodes
    top_nodes = [None]
    with torch.no_grad():
        while num_tokens_gen < num_tokens:
            out, state = net(input, state)  # (?,batch,vocab_size), (batch,hidden)
            # If this is the prediction for the beginning tokens, keep only last
            # time step
            out = out[-1, :, :]  # note: the first dimension is removed
            out = F.log_softmax(out, 1) # (batch,hidden)

            # Add every possible new nodes
            prev_nodes = top_nodes  # previous nodes
            nodes = list()
            for i, prev_node in enumerate(prev_nodes):

                log_probas = out[i, :]  # (vocab_size,)
                assert(log_probas.size() == (vocab.SIZE,))
                # for every possible token, add a new child node
                for token, logp in enumerate(log_probas):
                    node = BeamSearchNode(tuple(tensor[:, i, :] for tensor in state),
                                          prev_node, token, logp)
                    nodes.append(node)

            # print("len(nodes):", len(nodes))
            # print("len(prev_nodes)*vocab_size:", len(prev_nodes)*vocab.SIZE)

            # Get the top-k nodes
            # top_nodes_h = heapq.nlargest(top_k, nodes)
            #@bug: top_nodes_h is not always equal to top_nodes, especially at the last
            # time steps!!! This is weird, maybe ignore and use sorted() anyway

            top_nodes = sorted(nodes, reverse=True)[:top_k]
            # extract the top k predicted tokens in a tensor
            # the top k tokens are treated as a batch of input of size (1, top_k)
            input = torch.tensor([node.token for node in top_nodes]).view(1, -1).to(
                input.device)
            # extract the states for the top k predicted token
            state = tuple(torch.stack(tensors, 1)
                          for tensors in zip(*(node.rnn_state for node in top_nodes)))
            num_tokens_gen += 1

    # extract top k generated texts
    gens = list()
    for last_node in top_nodes:
        gen = list()
        node = last_node
        while node.prev_node is not None:
            gen.append(node.token)
            node = node.prev_node
        gen.reverse()
        gens.append(gen)
    texts = [vocab.code2string(gen) for gen in gens]
    return texts

# def generate_tokens_beam_search(net, beginning, num_tokens, top_k, string2code, code2string, eos_code):
#     """Generate characters using beam search"""
#     net.eval()

#     input = string2code(beginning).view(-1, 1)  # input
#     gens = list()
#     num_tokens_gen = 0
#     state = 0
#     with torch.no_grad():
#         while num_tokens_gen < num_tokens:
#             out, state = net(input, state)
#             print('Raw out size:', out.size())
#             print('Hidden size:', state[0].size())
#             # If the predictions are for the beginning tokens, get only
#             # prediction at last time step
#             out = out[-1, :, :]  # note: the first dimension is removed
#             out = F.log_softmax(out, 1)
#             # extract the top k predicted tokens
#             top_out, top_indices = torch.topk(out, top_k, 1)
#             # top k are treated inputed in parallel, like a batch of size (1, top_k)
#             print('Top indices:', top_indices.size(), top_indices)
#             input = top_indices
#             # repeat the hidden and memory states to match top_indices dimension 1
#             if num_tokens_gen == 0:
#                 state = tuple(tensor.repeat(top_k, 1) for tensor in state)
#             print('Hidden repeated:', state[0].size())
#             num_tokens_gen += 1

#     text = beginning + code2string(gens)
#     return text
