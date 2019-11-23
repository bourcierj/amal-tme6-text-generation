"""Dataset utilities."""

from pathlib import Path
import re
import string
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TrumpVocabulary:
    """Helper class to manage a vocabulary of characters."""

    # the vocabulary of letters
    LETTERS = string.ascii_letters + string.punctuation + string.digits + ' ' + '—' + '—'
    id2letter = dict(zip(range(1, len(LETTERS) + 1), LETTERS))
    id2letter[0] = ''  # Null character
    SIZE = len(id2letter)
    # The end_of_sequence character is set to '#' (code 55)
    letter2id = dict(zip(id2letter.values(), id2letter.keys()))
    EOS_ID = letter2id['#']

    @classmethod
    def normalize(cls, text: str):
        """Normalizes a text"""
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                       if c in cls.LETTERS)

    @classmethod
    def string2code(cls, sen: str):
        """Encodes a sentence."""
        return torch.tensor([cls.letter2id[c] for c in sen] + [cls.EOS_ID])

    @classmethod
    def code2string(cls, t: torch.Tensor, include_eos=True):
        """Decodes a string"""
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        if include_eos:
            return ''.join(cls.id2letter[i] for i in t)

        return ''.join(cls.id2letter[i] for i in t if i != cls.EOS_ID)


def process(datapath):
    """Processes the Trump speeches.
    The tokenizing used is character-level.
    Returns:
        list: list of speeches. Each speech is a list of sentences, and each
        sentence is a list of integer-encoded chars.
    """
    vocab = TrumpVocabulary()
    RE_SENTENCE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

    speeches2code = list()
    with open(datapath, 'r') as fp:
        for line in fp:
            line = line.rstrip().replace('\x96', '—').replace('\x97', '—')\
                       .replace('\x85', '...')
            line = vocab.normalize(line)
            # get sentences
            sentences = re.split(RE_SENTENCE, line)
            # encode sentences as integers
            coded = [vocab.string2code(s) for s in sentences]
            speeches2code.append(coded)

    return speeches2code


class TrumpDataset(Dataset):
    """Dataset for Trump speeches."""

    def __init__(self, datapath):

        speeches = process(datapath)  # extract processed speeches
        # record speech ids for every sentence
        self.speech_ids = [id for (id, speech) in enumerate(speeches) for s in speech]
        # get all sentences
        self.sentences = [s for speech in speeches for s in speech]
        # codes of tokens beginning sentences: useful for generating text
        # self.begin_codes = set(s[0].item() for s in self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    @staticmethod
    def collate(batch):
        """Collate function (pass to DataLoader's collate_fn arg).
        Args:
            batch (list): list of examples returned by __getitem__
        Returns:
            tuple: Pair of tensors: the batch of padded sequences, and the targets
                (ie the tokens at next time step)
        """
        # pad sequences with 0
        text = pad_sequence(batch)
        # define targets, ie the tokens at next time steps
        target = torch.zeros_like(text)
        target[:-1, :] = text[1:, :]
        return (text, target)

def get_dataloader(batch_size):

    datapath = Path('../tme4-rnn/data/trump_full_speech.txt')
    dataset = TrumpDataset(datapath)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=dataset.collate,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=torch.multiprocessing.cpu_count())
    return loader


if __name__ == '__main__':

    vocab = TrumpVocabulary()
    print("LETTERS: ", vocab.LETTERS, '\n')
    print("Vocabulary size:", vocab.SIZE)

    print("id2letter:\n", vocab.id2letter, '\n')
    datapath = Path('../tme4-rnn/data/trump_full_speech.txt')
    speeches2code = process(datapath)
    dataset = TrumpDataset(datapath)

    print("Trump speeches dataset summary:\n")
    print(f"Number of speeches: {len(speeches2code)}")
    print(f"Number of sentences: {len(dataset)}")
    print("Number of characters (including EOS): "
          f"{sum(s.numel() for s in dataset.sentences)}\n")

    print("Train samples:\n")
    for i in range(128, 136):
        input = dataset[i]
        print(f'Input: {vocab.code2string(input)}')

    #print("Tokens beggining sentences:", set(id2letter[c] for c in dataset.begin_codes))
    # batch = dataset.sentences[:8]
    # data, target = dataset.collate(batch)
    # print(f"Example of batch:\n Data: {tuple(data.size())}\n", data)
    # print(f"Target: {tuple(target.size())}\n",target)
