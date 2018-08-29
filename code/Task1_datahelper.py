import os
import csv
import numpy as np
import shelve
import gensim
import numpy as np
import re
from time import time
import pickle

LIMIT_LENGTH_OF_SENTENCES = 300

def readFileTSV(dirname='../train_tsv'):
    sentences = []
    labels = []
    sequence_lengths = []
    sent = []
    label = []
    leng = 0

    for filename in os.listdir(dirname):
        with open(os.path.join(dirname, filename)) as f:
            for _, line in enumerate(f.readlines()):
                row = line.strip()
                row = re.split("\t", row)
                if len(row) == 1:
                    if leng > 0 and leng <= LIMIT_LENGTH_OF_SENTENCES:
                        sentences.append(sent)
                        labels.append(label)
                        sequence_lengths.append(leng)
                    leng = 0
                    label = []
                    sent = []
                    
                elif len(row) == 3:
                    sent.append(row[0].lower().strip())
                    label.append(row[1])
                    leng += 1

    return np.array(sentences), np.array(labels), np.array(sequence_lengths)

def get_vocabs(sentences, sequence_lengths, max_doc_len):
    vocabs = []
    n_samples = len(sentences)
    for idx in range(n_samples):
        sentence = sentences[idx]
        for subidx in range(min(max_doc_len, sequence_lengths[idx])):
            if sentence[subidx] not in vocabs:
                vocabs.append(sentence[subidx])
    vocabs.append('')
    return vocabs

def generate_char_dict():
    char_dict = {}
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{}'
    for i,c in enumerate(alphabet):
        char_dict[c] = i
    return char_dict

def word_2_indices_per_char(word, max_word_len, char_dict):
    data = np.zeros(max_word_len, dtype=np.int32)
    rest = max_word_len - len(word)
    for i in range(0, len(word)):
        if i >= max_word_len:
            break
        elif word[i] in char_dict:
            data[i + rest//2] = char_dict[word[i]]
        else:
            # unknown character set to be 0
            data[i + rest//2] = 0
    return data

def generate_lookup_word_embedding(vocabs, filename='../bio-domain word2vec/PubMed-shuffle-win-30.bin'):
    pretrained_word2vec = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    dims = len(pretrained_word2vec[[*pretrained_word2vec.vocab.keys()][0]])
    n_vocabs = len(vocabs)

    lookup_table = np.zeros(shape=[n_vocabs, dims])
    index_of_word_in_lookup_table = dict()

    for idx, word in enumerate(vocabs):
        assert word not in index_of_word_in_lookup_table
        index_of_word_in_lookup_table[word] = int(idx)
        if word in pretrained_word2vec:
            lookup_table[idx, :] = pretrained_word2vec[word]
        else:
            lookup_table[idx, :] = np.random.randn(dims)
    return lookup_table, index_of_word_in_lookup_table

def encode_sentences(sentences, max_doc_len, lookup_table, index_of_word_in_lookup_table):
    encoded_sentences = np.zeros(shape=[sentences.shape[0], max_doc_len], dtype=np.int32)
    for idx_sent, sent in enumerate(sentences):
        for idx_word_in_sent in range(max_doc_len):
            word = sent[idx_word_in_sent]
            idx_of_word = index_of_word_in_lookup_table[word]
            encoded_sentences[idx_sent,idx_word_in_sent] = idx_of_word
    return encoded_sentences
    
def train_dev_split(sentences, labels, sequence_lengths, train_ratio=0.8):
    n_samples = sentences.shape[0]
    n_train_samples = int(train_ratio*n_samples)
    
    train_idx = np.random.choice(n_samples, n_train_samples, replace=False)
    test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))

    train_sent, dev_sent = sentences[train_idx], sentences[test_idx]
    train_labels, dev_labels = labels[train_idx], labels[test_idx]
    train_sequence_lengths, dev_sequence_lengths = sequence_lengths[train_idx], sequence_lengths[test_idx]
    
    return {
        "train_sentences": train_sent,
        "dev_sentences": dev_sent,
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "train_sequence_lengths": train_sequence_lengths,
        "dev_sequence_lengths": dev_sequence_lengths
    }

def get_labels_template(labels):
    labels_template = []
    for label in labels:
        for sublabel in label:
            if sublabel not in labels_template:
                labels_template.append(sublabel)
    return labels_template

def encode_labels(labels, max_doc_len, labels_template):
    labels_encoded = np.zeros(shape=[labels.shape[0], max_doc_len], dtype=np.int32)
    for idx, label in enumerate(labels):
        for subidx in range(max_doc_len):
            sublabel = label[subidx]
            labels_encoded[idx, subidx] = labels_template.index(sublabel)
    return labels_encoded

def decode_labels(encoded_labels, labels_template):
    labels_decoded = []
    for label in encoded_labels:
        label_decoded = []
        for sublabel in label:
            label_decoded.append(labels_template[sublabel])
        labels_decoded.append(label_decoded)
    return np.array(labels_decoded)

def get_max_doc_len(sequence_lengths):
    max_doc_len = 0
    for leng in sequence_lengths:
        max_doc_len = max(max_doc_len, leng)
    return max_doc_len
def add_padding(sentences, labels, sequence_lengths, max_doc_len, dims=200):
    for idx, leng in enumerate(sequence_lengths):
        gap = LIMIT_LENGTH_OF_SENTENCES - leng
        sentences[idx].extend(['']*gap)
        labels[idx].extend(['PAD']*gap)
    

def batch_iter(sentences, labels, sequence_lengths, batch_size=32, num_epochs=1000, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    n_samples = sentences.shape[0]
    num_batches_per_epoch = int((n_samples-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(n_samples))

            sentences = sentences[shuffle_indices]
            labels = labels[shuffle_indices]
            sequence_lengths = sequence_lengths[shuffle_indices]
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, n_samples)
            yield sentences[start_index:end_index], labels[start_index:end_index], sequence_lengths[start_index:end_index]

def get_word_from_idx(index_of_word_in_lookup_table, idx):
    for word in index_of_word_in_lookup_table:
        if index_of_word_in_lookup_table[word] == int(idx):
            return word
    return ' '

def word_indices_to_char_indices(sents, lengths, max_doc_len, max_word_len, char_dict, index_of_word_in_lookup_table):
    batch_size = sents.shape[0]
    res = np.zeros(shape=[batch_size, max_doc_len, max_word_len], dtype=np.int32)
    for idx_sent, sentence in enumerate(sents):
        for idx_word, word_indices in enumerate(sentence):
            if idx_word < lengths[idx_sent]:
                word = get_word_from_idx(index_of_word_in_lookup_table, word_indices)
                char_indices = word_2_indices_per_char(word, max_word_len, char_dict)
                res[idx_sent, idx_word, :] = char_indices
            else:
                break
    return res.reshape((batch_size, max_doc_len*max_word_len))

def write_to_file(data, filename='../train_dev_data.shlv'):
    with shelve.open(filename) as f:
        f['labels_template'] = data['labels_template']
        f['max_doc_len'] = data['max_doc_len']
        f['word_embedding_lookup_table'] = data['word_embedding_lookup_table']
        f['index_of_word_in_lookup_table'] = data['index_of_word_in_lookup_table']
        f['max_word_len'] = data['max_word_len']

        f["train_sentences"] = data["train_sentences"]
        f["train_labels"] = data["train_labels"]
        f["train_sequence_lengths"] = data["train_sequence_lengths"]

        f["dev_sentences"] = data["dev_sentences"]
        f["dev_labels"] = data["dev_labels"]
        f["dev_sequence_lengths"] = data["dev_sequence_lengths"]
        f['char_dict'] = data['char_dict']
    
def load_from_file(filename='../train_dev_data.shlv'):
    data = dict()
    with shelve.open(filename) as f:
        data['labels_template'] = f['labels_template']
        data['max_doc_len'] = f['max_doc_len']
        data['word_embedding_lookup_table'] = f['word_embedding_lookup_table']
        data['index_of_word_in_lookup_table'] = f['index_of_word_in_lookup_table']
        data['max_word_len'] = f['max_word_len']

        data["train_sentences"] = f["train_sentences"]
        data["train_labels"] = f["train_labels"]
        data["train_sequence_lengths"] = f["train_sequence_lengths"]

        data["dev_sentences"] = f["dev_sentences"]
        data["dev_labels"] = f["dev_labels"]
        data["dev_sequence_lengths"] = f["dev_sequence_lengths"]
        data['char_dict'] = f['char_dict']

    return data


def next_lr(lr, p=0.05, t=180):
    return 1.0*lr/(1.0 + p*t)

if __name__ == '__main__':
    def preprocess():
        sentences, labels, sequence_lengths = readFileTSV()
        max_doc_len = get_max_doc_len(sequence_lengths)
        add_padding(sentences, labels, sequence_lengths, max_doc_len)

        #lấy toàn bộ label và unique, đưa vào labels_template
        labels_template = get_labels_template(labels)
        encoded_labels = encode_labels(labels, max_doc_len, labels_template) #chứa labels dạng số #cần tối ưu

        vocabs = get_vocabs(sentences, sequence_lengths, max_doc_len)
        lookup_table, index_of_word_in_lookup_table = generate_lookup_word_embedding(vocabs)
        encoded_sentences = encode_sentences(sentences, max_doc_len, lookup_table, index_of_word_in_lookup_table) #chứa sentence dạng số

        char_dict = generate_char_dict()

        data = train_dev_split(encoded_sentences, encoded_labels, sequence_lengths, train_ratio=0.8)
        data['labels_template'] = labels_template
        data['max_doc_len'] = max_doc_len
        data['word_embedding_lookup_table'] = lookup_table
        data['index_of_word_in_lookup_table'] = index_of_word_in_lookup_table
        data['max_word_len'] = 30
        data['char_dict'] = char_dict
        write_to_file(data)   

    data = load_from_file()

    max_doc_len = data['max_doc_len']
    labels_template = data['labels_template']

    train_sentences = data["train_sentences"]
    train_labels = data["train_labels"]
    train_sequence_lengths = data["train_sequence_lengths"]

    dev_sentences = data["dev_sentences"]
    dev_labels = data["dev_labels"]
    dev_sequence_lengths = data["dev_sequence_lengths"]

    word_lookup_table = data['word_embedding_lookup_table']
    index_of_word_in_lookup_table = data['index_of_word_in_lookup_table']

    with open('data_train.tsv', 'w') as f:
        for idx in range(len(train_sequence_lengths)):
            for subidx in range(train_sequence_lengths[idx]):
                word = get_word_from_idx(index_of_word_in_lookup_table, train_sentences[idx][subidx])
                label = labels_template[train_labels[idx][subidx]]
                f.write("%s\t%s\n" % (word, label))
                
    with open('data_dev.tsv', 'w') as f:
        for idx in range(len(dev_sequence_lengths)):
            for subidx in range(dev_sequence_lengths[idx]):
                word = get_word_from_idx(index_of_word_in_lookup_table, dev_sentences[idx][subidx])
                label = labels_template[dev_labels[idx][subidx]]
                f.write("%s\t%s\n" % (word, label))
