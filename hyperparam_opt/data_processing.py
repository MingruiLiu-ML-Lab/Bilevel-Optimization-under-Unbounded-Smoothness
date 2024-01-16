import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import json, pickle
GLOVE_DIM = 300
LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}

class data_loader(Dataset):

    def __init__(self, examples, num_task, k_support, k_query, word2vec):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        random.shuffle(self.examples)
        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.max_seq_length = 256
        self.labels = ["positive", "negative"]
        labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Read sentence and label data for current split from files.
        s1_path = "../data/news-data/dataset.json"

        self.s1_sentences = [line.rstrip() for line in open(s1_path, "r")]

        self.dataset_size = len(self.s1_sentences)

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = "../data/news-data/vocab.pkl"
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
            for line in open(s1_path, "r"):
                for word in line.rstrip().split():
                    if word not in vocab:
                        vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word2vec = {}
        glove_path = "../data/news-data/glove.840B.300d.txt"
        wordvec_path = "../data/news-data/wordvec.pkl"
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word2vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            with open(glove_path, "r", encoding='utf-8') as glove_file:
                for line in glove_file:
                    word, vec = line.split(' ', 1)
                    if word in vocab:
                        self.word2vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word2vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word2vec
        assert "</s>" in self.word2vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word2vec] +
                ["</s>"]
            )

        self.create_batch(self.num_task)

    def create_batch(self, num_task):
        self.supports = []  # support set
        self.queries = []  # query set

        for b in range(num_task):
            # 1.select domain randomly
            domain = random.choice(self.examples)['domain']
            d_examples = [e for e in self.examples if e['domain'] == domain]

            domainExamples = self.split_words(d_examples)
            inds = torch.randperm(len(domainExamples[1]))[:self.k_support + self.k_query]
            # 1.select k_support + k_query examples from domain randomly
            selected_examples = np.array(domainExamples[0])[inds.numpy()]
            selected_examples_labels = np.array(domainExamples[1])[inds.numpy()]
            selected_examples_vec = self.word_vec(selected_examples)
            # selected_examples_vec = selected_examples
            exam_train = selected_examples_vec[:self.k_support]
            exam_train_label = selected_examples_labels[:self.k_support]
            exam_test = selected_examples_vec[self.k_support:]
            exam_test_label = selected_examples_labels[self.k_support:]

            self.supports.append((exam_train, exam_train_label))
            self.queries.append((exam_test, exam_test_label))

    def split_words(self, examples):
        label_map ={'positive':0, 'negative':1}
        sentences = []
        targets =[]
        for i in range(len(examples)):
            sent = examples[i]['text']
            targets.append(label_map[examples[i]['label']])
            sentences.append(np.array(
                [word for word in sent.split() if word in self.word2vec]
            ))
        return (sentences, targets)
    def word_vec(self,examples):
        s1_embed = [None for _ in range(len(examples))]
        for i in range(len(examples)):
            s1_embed[i] = np.zeros((len(examples[i]), GLOVE_DIM))
            for j in range(len(examples[i])):
                try:
                    s1_embed[i][j]=self.word2vec[examples[i][j]]
                except:
                    pass

        return s1_embed

    def __getitem__(self, index):
        support_set = self.supports[index]
        query_set   = self.queries[index]
        return support_set, query_set

    def __len__(self):
        return self.num_task