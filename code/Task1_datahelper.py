
import os
import csv
import numpy as np
import shelve
import gensim
import numpy as np
import re
from convertXLStoXLSX import writeToExcelXLSX

LIMIT_LENGTH_OF_SENTENCES = 100

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
                if len(row) == 1 or row[1] == '*':
                    if leng <= LIMIT_LENGTH_OF_SENTENCES:
                        sentences.append(sent)
                        labels.append(label)
                        sequence_lengths.append(leng)
                    leng = 0
                    label = []
                    sent = []
                    
                elif len(row) == 3:
                    sent.append(row[0].lower())
                    label.append(row[1])
                    leng += 1
    # for d in set(db):
    #     print(repr(d))
    return sentences, labels, sequence_lengths

def convert_word_to_vec(sentences, sequence_lengths, filename='../bio-domain word2vec/PubMed-shuffle-win-30.bin'):
    vectors = []
    nrow = len(sentences)
    ncol = len(sentences[0])
    for idx in range(nrow):
        vectors.append([None]*ncol)
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    cnt = 0
    total = 0
    for idx in range(nrow):
        sentence = sentences[idx]
        for subidx in range(ncol):
            if subidx < sequence_lengths[idx]:
                total += 1
            if sentence[subidx] in model and subidx < sequence_lengths[idx]:
                cnt += 1
                vectors[idx][subidx] = model[sentence[subidx]]
    print("Number of words in word2vec: %d in total %d words" % (cnt, total))
    
    dims = len(model[[*model.vocab.keys()][0]])
    del model

    random_word2vec = dict()
    for idx in range(nrow):
        sentence = sentences[idx]
        for subidx in range(ncol):
            if vectors[idx][subidx] is None:
                if sentences[idx][subidx] not in random_word2vec:
                    random_word2vec[sentences[idx][subidx]] = np.random.randn(dims)
                vectors[idx][subidx] = random_word2vec[sentences[idx][subidx]]

    return vectors

def train_dev_split(sentences, vectors, labels, sequence_lengths):
    n_samples = len(vectors)
    n_train_samples = int(0.9*n_samples)
    
    train_vectors, dev_vectors = vectors[:n_train_samples], vectors[n_train_samples:]
    train_labels, dev_labels = labels[:n_train_samples], labels[n_train_samples:]
    train_sequence_lengths, dev_sequence_lengths = sequence_lengths[:n_train_samples], sequence_lengths[n_train_samples:]
    train_sent, dev_sent = sentences[:n_train_samples], sentences[n_train_samples:]
    return {
        "train_sentences": train_sent,
        "dev_sentences": dev_sent,
        "train_vectors": train_vectors,
        "dev_vectors": dev_vectors,
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "train_sequence_lengths": train_sequence_lengths,
        "dev_sequence_lengths": dev_sequence_lengths
    }

def labels_sequence_to_int(labels):
    labels_template = []
    for label in labels:
        for sublabel in label:
            if sublabel not in labels_template:
                labels_template.append(sublabel)
    labels_int = []
    for label in labels:
        label_int = []
        for sublabel in label:
            label_int.append(labels_template.index(sublabel))
        labels_int.append(label_int)
    return labels_int, labels_template
def add_padding(sentences, labels, sequence_lengths, dims=200):
    max_doc_len = 0
    for leng in sequence_lengths:
        max_doc_len = max(max_doc_len, leng)
    for idx, leng in enumerate(sequence_lengths):
        gap = LIMIT_LENGTH_OF_SENTENCES - leng
        sentences[idx].extend([' ']*gap)
        labels[idx].extend(['PAD']*gap)
    return max_doc_len

def preprocess():
    sentences, labels, sequence_lengths = readFileTSV()
    max_doc_len = add_padding(sentences, labels, sequence_lengths)
    vectors = convert_word_to_vec(sentences, sequence_lengths)
    
    
    labels_int, labels_template = labels_sequence_to_int(labels)

    data = train_dev_split(sentences, vectors, labels_int, sequence_lengths)
    data['labels_template'] = labels_template
    data['max_doc_len'] = max_doc_len
    return data    

if __name__ == '__main__':
    sents, _, _ = readFileTSV()
    print('Total length:', len(sents))
    
    
