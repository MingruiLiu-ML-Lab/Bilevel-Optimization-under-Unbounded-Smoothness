import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import tqdm
import json, pickle
from torch.utils.data import TensorDataset
GLOVE_DIM = 300
GLOVE_NAME = "glove.840B.300d.txt"
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"

class SNLIDataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train", noise_rate=0):
        """ Initialize SNLI dataset. """

        assert split in ["train", "dev", "test"]
        self.root = os.path.join(root, "snli_1.0")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.n_classes = 3
        self.noise_rate = noise_rate

        """ Read and store data from files. """
        self.labels = ["entailment", "neutral", "contradiction"]
        labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Read sentence and label data for current split from files.
        s1_path = os.path.join(self.root, "SNLI", f"s1.{self.split}")
        s2_path = os.path.join(self.root, "SNLI", f"s2.{self.split}")
        target_path = os.path.join(self.root, "SNLI", f"labels.{self.split}")
        self.s1_sentences = [line.rstrip() for line in open(s1_path, "r")]
        self.s2_sentences = [line.rstrip() for line in open(s2_path, "r")]
        self.targets = np.array(
            [labels_to_idx[line.rstrip("\n")] for line in open(target_path, "r")]
        )
        assert len(self.s1_sentences) == len(self.s2_sentences)
        assert len(self.s1_sentences) == len(self.targets)
        self.dataset_size = len(self.s1_sentences)
        print(f"Loaded {self.dataset_size} sentence pairs for {self.split} split.")

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, "SNLI", VOCAB_NAME)
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "dev", "test"]:
                paths = [
                    os.path.join(self.root, "SNLI", f"s1.{split}"),
                    os.path.join(self.root, "SNLI", f"s2.{split}"),
                ]
                for path in paths:
                    for line in open(path, "r"):
                        for word in line.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, "GloVe", GLOVE_NAME)
        wordvec_path = os.path.join(self.root, "SNLI", WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    word, vec = line.split(' ', 1)
                    if word in vocab:
                        self.word_vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )
        for i in range(len(self.s2_sentences)):
            sent = self.s2_sentences[i]
            self.s2_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """ Return a single element of the dataset. """

        # Encode sentences as sequence of glove vectors.
        sent1 = self.s1_sentences[idx]
        sent2 = self.s2_sentences[idx]
        s1_embed = np.zeros((len(sent1), GLOVE_DIM))
        s2_embed = np.zeros((len(sent2), GLOVE_DIM))
        for j in range(len(sent1)):
            s1_embed[j] = self.word_vec[sent1[j]]
        for j in range(len(sent2)):
            s2_embed[j] = self.word_vec[sent2[j]]
        s1_embed = torch.from_numpy(s1_embed).float()
        s2_embed = torch.from_numpy(s2_embed).float()

        # add noise_rate
        if np.random.rand(1) <= self.noise_rate:
            # Convert targets to tensor.
            target = torch.tensor([int(self.targets[idx]+ np.random.randint(1, self.n_classes))%3]).long()
        else:
            # Convert targets to tensor.
            target = torch.tensor([self.targets[idx]]).long()
        return (s1_embed, s2_embed), target, idx

    @property
    def n_words(self):
        return len(self.word_vec)


def collate_pad_double(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """

    s1_embeds = [x[0][0] for x in data_points]
    s2_embeds = [x[0][1] for x in data_points]
    targets = [x[1] for x in data_points]
    inds = [x[2] for x in data_points]
    # Get sentences for batch and their lengths.
    s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
    max_s1_len = np.max(s1_lens)
    s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
    max_s2_len = np.max(s2_lens)
    lens = (s1_lens, s2_lens)

    # Encode sentences as glove vectors.
    bs = len(data_points)
    s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
    s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
    for i in range(bs):
        e1 = s1_embeds[i]
        e2 = s2_embeds[i]
        s1_embed[: len(e1), i] = e1.clone()
        s2_embed[: len(e2), i] = e2.clone()
    embeds = (
        torch.from_numpy(s1_embed).float(), torch.from_numpy(s2_embed).float()
    )

    # Convert targets to tensor.
    targets = torch.cat(targets)

    return (embeds, lens), targets, inds

class Sent140Dataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train", noise_rate=0):
        """ Initialize Sent140 dataset. """

        assert split in ["train", "test"]
        self.root = os.path.join(root, "sent140")
        self.data_path = os.path.join(self.root, f"{split}.json")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.noise_rate = noise_rate
        self.n_classes = 2

        # Read sentence and label data for current split from file.
        with open(self.data_path, "r") as f:
            all_data = json.load(f)
        self.users = range(len(all_data["users"]))
        self.num_clients = len(self.users)
        self.sentences = []
        self.labels = []
        self.user_items = {}
        def process_label(l):
            if l == "0":
                return 0
            elif l == "4":
                return 1
            else:
                raise ValueError

        j = 0
        for i in self.users:
            user = all_data["users"][i]
            self.user_items[i] = []
            tweets = all_data["user_data"][user]["x"]
            labels = all_data["user_data"][user]["y"]
            assert len(tweets) == len(labels)
            for tweet_data, label in zip(tweets, labels):
                self.sentences.append(tweet_data[4])
                self.labels.append(process_label(label))
                self.user_items[i].append(j)
                j += 1

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, VOCAB_NAME)
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "test"]:
                path = os.path.join(self.root, f"{split}.json")
                with open(path, "r") as f:
                    split_data = json.load(f)
                split_sentences = split_data["user_data"]

                for user in tqdm(split_data["users"]):
                    for tweet_data in split_data["user_data"][user]["x"]:
                        sentence = tweet_data[4]
                        for word in sentence.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, GLOVE_NAME)
        wordvec_path = os.path.join(self.root, WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            lines = []
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    lines.append(line)
            for line in tqdm(lines):
                word, vec = line.split(' ', 1)
                if word in vocab:
                    self.word_vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = np.array(
                ["<s>"] +
                [word for word in sentence.split() if word in self.word_vec] +
                ["</s>"]
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """ Return a single element of the dataset. """

        # Encode sentence as sequence of glove vectors.
        sent = self.sentences[idx]
        sent_embed = np.zeros((len(sent), GLOVE_DIM))
        for j in range(len(sent)):
            sent_embed[j] = self.word_vec[sent[j]]
        sent_embed = torch.from_numpy(sent_embed).float()
        # add noise_rate
        if np.random.rand(1) <= self.noise_rate:
            # Convert targets to tensor.
            target = torch.tensor([int(self.labels[idx]+ np.random.randint(1, self.n_classes))%self.n_classes]).long()
        else:
            # Convert targets to tensor.
            target = torch.tensor([self.labels[idx]]).long()


        return sent_embed, target, idx

    @property
    def n_words(self):
        return len(self.word_vec)

def collate_pad(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """

    sent_embeds = [x[0] for x in data_points]
    targets = [x[1] for x in data_points]

    # Get sentences for batch and their lengths.
    lens = np.array([sent.shape[0] for sent in sent_embeds])
    max_sent_len = np.max(lens)
    inds = [x[2] for x in data_points]
    # Encode sentences as glove vectors.
    bs = len(data_points)
    sent_embed = np.zeros((max_sent_len, bs, GLOVE_DIM))
    for i in range(bs):
        e = sent_embeds[i]
        sent_embed[: len(e), i] = e.clone()
    sent_embed = torch.from_numpy(sent_embed).float()

    # Convert targets to tensor.
    targets = torch.cat(targets)

    return (sent_embed, lens), targets, inds

