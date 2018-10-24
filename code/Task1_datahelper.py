import os
import csv
import numpy as np
import shelve

import numpy as np
import re
from time import time
from math import sqrt

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

def get_idx_train_dev(sentences, train_ratio):
    n_samples = sentences.shape[0]

    n_train_samples = int(train_ratio*n_samples)
    
    train_idx = np.random.choice(n_samples, n_train_samples, replace=False)
    test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))
    return train_idx, test_idx

def get_vocabs(sentences):
    vocabs = []
    for sent in sentences:
        for word in sent:
            if word not in vocabs:
                vocabs.append(word)
    vocabs.append('')
    return vocabs

def get_map_word_id_and_map_id_word(vocabs):
    map_word_id = dict()
    map_id_word = dict()
    for id_, word in enumerate(vocabs):
        map_id_word[id_] = word
        map_word_id[word] = id_
    return map_word_id, map_id_word

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

def generate_lookup_word_embedding(vocabs, map_word_id, type_embeddings='word2vec'):
    pretrained_word2vec = None
    if type_embeddings == 'bio-word2vec':
        from gensim.models.keyedvectors import KeyedVectors
        filename = '../embeddings/wikipedia-pubmed-and-PMC-w2v.bin'
        pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
    elif type_embeddings == 'bio-word2vec-old':
        from gensim.models.keyedvectors import KeyedVectors
        filename = '../embeddings/PubMed-shuffle-win-2.bin'
        pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
    elif type_embeddings == 'word2vec':
        from gensim.models.keyedvectors import KeyedVectors
        filename = '../embeddings/GoogleNews-vectors-negative300.bin'
        pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)

    elif type_embeddings == 'fasttext':
        from gensim.models.keyedvectors import KeyedVectors
        filename = '../embeddings/crawl-300d-2M-subword.vec'
        pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=False)

    elif type_embeddings == 'glove':
        filename = '../embeddings/glove.6B.100d.txt'
        from gensim.scripts.glove2word2vec import glove2word2vec
        from gensim.test.utils import get_tmpfile
        from gensim.models.keyedvectors import KeyedVectors
        tmp_file = get_tmpfile("glove_to_word2vec.txt")
        glove2word2vec(filename, tmp_file)
        pretrained_word2vec = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
    elif type_embeddings == 'random':
        pretrained_word2vec = dict()
        dims = 300
        pretrained_word2vec[''] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)
    #print(type_embeddings, end=' ')
    assert pretrained_word2vec is not None
    if type_embeddings != 'random':
        dims = len(pretrained_word2vec[[*pretrained_word2vec.vocab.keys()][0]])
    else: 
        dims = 300
    n_vocabs = len(vocabs)

    OOV = []
    lookup_table = np.zeros(shape=[n_vocabs, dims])
    c = 0
    for _, word in enumerate(vocabs):
        idx = map_word_id[word]

        if word in pretrained_word2vec:
            lookup_table[idx, :] = pretrained_word2vec[word]
        else:
            c += 1
            lookup_table[idx, :] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)
            OOV.append(word)

    for k in OOV_dict:
        print("%s\t%d" % (k, OOV_dict[k]))
    # print(c)
    OOV.sort()
    with open('oov_dict.txt', 'w') as f:
        for line in OOV:
            f.write("%s\n" % line)
    return lookup_table

def encode_sentences(sentences, map_word_id):
    encoded_sentences = np.zeros(shape=[sentences.shape[0]], dtype=object)
    for idx, sent in enumerate(sentences):
        tmp = np.zeros(shape=[len(sent)], dtype=np.int32)
        for subidx, word in enumerate(sent):
            idx_of_word = map_word_id[word]
            tmp[subidx] = idx_of_word
        encoded_sentences[idx] = tmp
    return encoded_sentences
    
def train_dev_split(sentences, labels, sequence_lengths, train_idx, test_idx, type_embeddings, tagging):
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
    labels_template.append('PAD')
    return labels_template

def encode_labels(labels, labels_template):
    labels_encoded = np.zeros(shape=[labels.shape[0]], dtype=object)
    for idx, label in enumerate(labels):
        tmp = np.zeros(shape=[len(label)], dtype=np.int32)
        for subidx, sublabel in enumerate(label):
            tmp[subidx] = labels_template.index(sublabel)
        labels_encoded[idx] = tmp
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

    
    
def batch_iter(sentences, labels, sequence_lengths, idx_of_word_pad, idx_of_label_pad, batch_size=32, num_epochs=1000, shuffle=True):
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

            max_sentences_length_in_batch = max([length for length in sequence_lengths[start_index:end_index]])

            actual_batch_size = min(batch_size, end_index - start_index)
            padded_sentences = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)
            padded_labels = np.zeros(shape=[actual_batch_size, max_sentences_length_in_batch], dtype=np.int32)

            # print(start_index, ' ', end_index, ' ', end_index - start_index)
            for idx in range(min(batch_size, end_index - start_index)):
                for subidx in range(max_sentences_length_in_batch):
                    if subidx < sequence_lengths[start_index + idx]:
                        padded_sentences[idx][subidx] = sentences[start_index + idx][subidx]
                        padded_labels[idx][subidx] = labels[start_index + idx][subidx]
                        
                    else:
                        padded_sentences[idx][subidx] = idx_of_word_pad
                        padded_labels[idx][subidx] = idx_of_label_pad
                    

            yield padded_sentences, padded_labels, sequence_lengths[start_index:end_index], max_sentences_length_in_batch


def get_word_from_idx(map_id_word, idx):
    return map_id_word[idx]

def word_indices_to_char_indices(sents, lengths, max_doc_len, max_word_len, char_dict, map_id_word):
    batch_size = sents.shape[0]
    res = np.zeros(shape=[batch_size, max_doc_len, max_word_len], dtype=np.int32)
    for idx_sent, sentence in enumerate(sents):
        for idx_word, word_indices in enumerate(sentence):
            if idx_word < lengths[idx_sent]:
                word = get_word_from_idx(map_id_word, word_indices)
                char_indices = word_2_indices_per_char(word, max_word_len, char_dict)
                res[idx_sent, idx_word, :] = char_indices
            else:
                break
    return res

def write_to_file(data, filename):
    with shelve.open(filename) as f:
        for k in data:
            f[k] = data[k]
    
def load_from_file(filename, metadata = '../data_feed_model/metadata.shlv'):
    data = dict()
    with shelve.open(filename) as f:
        for k in f:
            data[k] = f[k]
    with shelve.open(metadata) as f:
        for k in f:
            data[k] = f[k]
    return data


def next_lr(lr, p=0.05, t=100):
    return 1.0*lr/(1.0 + p*t)

def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def update_tag_scheme(labels, sequence_lengths, tag_scheme='BIOES'):
    """
    Check and update sentences tagging scheme to BIO2
    Only BIO1 and BIO2 schemes are accepted for input data.
    """
    for i, label in enumerate(labels):

        tags = []
        for subi in range(sequence_lengths[i]):
            tags.append(label[subi])
        # Check that tags are given in the BIO format
        if not iob2(tags):
            print(tags)
            raise Exception('Sentences should be given in BIO format! ')
        if tag_scheme == 'BIOES':
            new_tags = iob_iobes(tags)
            for subidx, new_tag in enumerate(new_tags):
                label[subidx] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')

def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags



def update_lookup_table_for_testing(test_sentences, lookup_table, map_id_word, map_word_id, type_embeddings):
    n_out_of_vocabs = 0
    out_of_vocabs = list()
    old_n_vocabs = len(map_word_id.keys())
    oov_map_word_id = dict()
    for sent in test_sentences:
        for word in sent:
            if word not in map_word_id:
                oov_map_word_id[word] = n_out_of_vocabs
                n_out_of_vocabs += 1
                out_of_vocabs.append(word)

    #update map_id_word and map_word_id
    for idx, word in enumerate(out_of_vocabs):
        map_word_id[word] = idx + old_n_vocabs
        map_id_word[idx + old_n_vocabs] = word

    #update lookup table
    dims = lookup_table.shape[1]
    updated_lookup_table = np.zeros(shape=[lookup_table.shape[0] + n_out_of_vocabs, dims])

    oov_lookup_table = generate_lookup_word_embedding(out_of_vocabs, oov_map_word_id, type_embeddings=type_embeddings)
    for idx in range(lookup_table.shape[0]):
        updated_lookup_table[idx, :] = lookup_table[idx, :]

    for idx, word in enumerate(oov_lookup_table):
        updated_lookup_table[idx + old_n_vocabs, :] = oov_lookup_table[idx, :]

    return updated_lookup_table


def get_feed_dict_for_testting(test_sentences, test_labels, test_sequence_lengths, max_word_len, char_dict, labels_template, updated_lookup_table, map_id_word, map_word_id, type_embeddings, tagging):

    dims = updated_lookup_table.shape[1]

    encoded_test_labels = encode_labels(test_labels, labels_template)
    encoded_test_sentences = encode_sentences(test_sentences, map_word_id)

    for idx in range(test_sentences.shape[0]):
        
        sent = encoded_test_sentences[idx].reshape(1, -1)
        label = encoded_test_labels[idx].reshape(1, -1)
        sequence_length = np.array([test_sequence_lengths[idx]])

        max_sentences_length_in_batch = sent.shape[1]

        vectors = np.zeros(shape=(1, max_sentences_length_in_batch, dims), dtype=np.float32)

        for subidx, word in enumerate(test_sentences[idx]):
            id_of_word = map_word_id[word]
            vectors[0, subidx, :] = updated_lookup_table[id_of_word, :]

        chars_indices = word_indices_to_char_indices(sent, sequence_length, max_sentences_length_in_batch, max_word_len, char_dict, map_id_word)
        feed_dict = {
                "labels_placeholder": label,
                "vectors": vectors,
                "sequence_lengths_placeholder": sequence_length,
                "chars_placeholder": chars_indices,
                "max_sentences_length_placeholder": max_sentences_length_in_batch,
                "dropout_prob_placeholder": 1.0
                }

        yield feed_dict
    


if __name__ == '__main__':
    def init_metadata():
        
        sentences, labels, sequence_lengths = readFileTSV()
        # max_doc_len = get_max_doc_len(sequence_lengths)
        # add_padding(sentences, labels, sequence_lengths, max_doc_len)
        vocabs = get_vocabs(sentences)
        
        train_idx, dev_idx = get_idx_train_dev(sentences,train_ratio=0.8)
        map_word_id, map_id_word = get_map_word_id_and_map_id_word(vocabs)
        for w in vocabs:
            assert w in map_word_id
        char_dict = generate_char_dict()
        
        with shelve.open('../data_feed_model/metadata.shlv') as f:
            f['sentences'] = sentences
            f['labels'] = labels
            f['sequence_lengths'] = sequence_lengths

            f['vocabs'] = vocabs
            f['map_word_id']= map_word_id
            f['map_id_word'] = map_id_word
            f['train_idx'] = train_idx
            f['dev_idx'] = dev_idx
            f['char_dict'] = char_dict
            f['max_word_len'] = 30
    def preprocess(type_embeddings, tagging):
        
        with shelve.open('../data_feed_model/metadata.shlv') as f:
            sentences = f['sentences']
            labels = f['labels']
            sequence_lengths = f['sequence_lengths']
            vocabs = f['vocabs']
            map_word_id = f['map_word_id']
            train_idx = f['train_idx'] 
            dev_idx = f['dev_idx']
        

        
        labels_updated = np.empty_like(labels)
        labels_updated[:] = labels


        if tagging == 'BIO':
            lookup_table = generate_lookup_word_embedding(vocabs, map_word_id, type_embeddings=type_embeddings)
        else: #BIOES
            with shelve.open("../data_feed_model/%s_BIO.shlv" % (type_embeddings)) as f:
                lookup_table = f['lookup_table']
            update_tag_scheme(labels_updated, sequence_lengths)
        labels_template = get_labels_template(labels)
        encoded_labels = encode_labels(labels_updated, labels_template) #chứa labels dạng số 
        encoded_sentences = encode_sentences(sentences, map_word_id)

        #print(encoded_labels)
        #print(encoded_sentences.shape)
        #print(encoded_labels.shape)

        data = train_dev_split(encoded_sentences, encoded_labels, sequence_lengths, train_idx, dev_idx, type_embeddings, tagging)
        
        data['labels_template'] = labels_template
        data['lookup_table'] = lookup_table
        write_to_file(data, "../data_feed_model/%s_%s.shlv" % (type_embeddings, tagging))


    # with shelve.open('../data_feed_model/metadata.shlv') as f:
    #     print(f['sentences'].shape[0])
    # init_metadata()
    OOV_dict = dict()
    for type_embeddings in ['bio-word2vec']:
        for tagging in ['BIO']:
            preprocess(type_embeddings, tagging)

    # data = load_from_file('../data_feed_model/bio-word2vec_BIO.shlv')
    # print(data["train_sentences"].shape)
    # print(data["dev_sentences"].shape)

    # from gensim.models.keyedvectors import KeyedVectors
    # filename = '../embeddings/wikipedia-pubmed-and-PMC-w2v.bin'
    # pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)

    # print('\'s' in pretrained_word2vec)
    # print('%' in pretrained_word2vec)
