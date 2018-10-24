import os
import shelve
import numpy as np
import tensorflow as tf
from time import time
from Task1_datahelper import load_from_file, word_2_indices_per_char, decode_labels, batch_iter, get_word_from_idx, word_indices_to_char_indices, next_lr, get_feed_dict_for_testting, readFileTSV, update_lookup_table_for_testing
from math import sqrt
import sys

type_embeddings = sys.argv[1].strip()
tagging = sys.argv[2].strip()

print()
print("Type embeddings: %s " % type_embeddings)
print("Tagging: %s" % tagging)

data = load_from_file("../data_feed_model/%s_%s.shlv" % (type_embeddings, tagging))

labels_template = data['labels_template']
lookup_table = data['lookup_table']
map_word_id = data['map_word_id']
map_id_word = data['map_id_word']
char_dict = data['char_dict']
max_word_len = data['max_word_len']

test_sentences, test_labels, test_sequence_lengths = readFileTSV('../test_tsv')

updated_lookup_table = update_lookup_table_for_testing(test_sentences, lookup_table, map_id_word, map_word_id, type_embeddings)

config = tf.ConfigProto(allow_soft_placement = True)

with tf.Session(config = config) as sess:
    saver = tf.train.import_meta_graph('../saved_model/%s-%s/ckpt.meta' % (type_embeddings, tagging))
    
    with tf.device("/device:gpu:0"):
        print(tf.train.latest_checkpoint('../saved_model/%s-%s/' % (type_embeddings, tagging)))
        saver.restore(sess, tf.train.latest_checkpoint('../saved_model/%s-%s/' % (type_embeddings, tagging)))
        graph = tf.get_default_graph()
        chars_placeholder = graph.get_tensor_by_name("characters:0")
        labels_placeholder = graph.get_tensor_by_name('labels:0')
        sequence_lengths_placeholder = graph.get_tensor_by_name('lengths:0')
        dropout_prob_placeholder = graph.get_tensor_by_name('dropout:0')
        max_sentences_length_in_batch = graph.get_tensor_by_name('max_sentences_length_in_batch:0')
        vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
        viterbi_sequence = graph.get_tensor_by_name("crf_decode/cond/Merge:0")
    
        tsvfile = open('../testing/predict_test_file_%s_%s.tsv' % (type_embeddings, tagging), 'w')

        feed_dicts = get_feed_dict_for_testting(test_sentences, test_labels, test_sequence_lengths, max_word_len, char_dict, labels_template, updated_lookup_table, map_id_word, map_word_id, type_embeddings, tagging)
        for idx, fd in enumerate(feed_dicts):
            # print('\n\n%s\n\n' % ' '.join(test_sentences[idx]))
            feed_dict = {
                chars_placeholder: fd['chars_placeholder'],
                labels_placeholder: fd['labels_placeholder'],
                sequence_lengths_placeholder: fd['sequence_lengths_placeholder'],
                dropout_prob_placeholder: fd['dropout_prob_placeholder'],
                vectors: fd['vectors'],
                max_sentences_length_in_batch: fd['sequence_lengths_placeholder'][0] 
            }

            predict = sess.run(viterbi_sequence, feed_dict=feed_dict)

            sent = test_sentences[idx]
            label = test_labels[idx]

            for subidx in range(test_sequence_lengths[idx]):        
                word = sent[subidx]
                golden_tag = label[subidx]
                predict_tag = labels_template[predict[0][subidx]]
                tsvfile.write("%s\t%s\t%s\n" % (word, golden_tag, predict_tag))
            tsvfile.write('-\tX\t-\n')

        tsvfile.close()


    

    

