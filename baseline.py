import os
import shelve
import numpy as np
import tensorflow as tf
from time import time
from Task1_datahelper import load_from_file, word_2_indices_per_char, decode_labels, batch_iter, get_word_from_idx
import sys
from math import sqrt

type_embeddings = sys.argv[1].strip()
tagging = sys.argv[2].strip()
print(type_embeddings)
print(tagging)
data = load_from_file("../data_feed_model/%s_%s.shlv" % (type_embeddings, tagging))

max_doc_len = data['max_doc_len']
max_word_len = data['max_word_len']
labels_template = data['labels_template']

train_sentences = data["train_sentences"]
train_labels = data["train_labels"]
train_sequence_lengths = data["train_sequence_lengths"]

dev_sents = data["dev_sentences"]
dev_labels = data["dev_labels"]
dev_sequence_lengths = data["dev_sequence_lengths"]

word_lookup_table = data['lookup_table']
map_word_id = data['map_word_id']
map_id_word = data['map_id_word']

num_classes = len(labels_template)
hidden_size_lstm = 200
dropout_prob = 0.5
n_epochs = 50
batch_size = 10
n_batches = int(train_sentences.shape[0]//batch_size) + 1
learning_rate = 0.015
momentum = 0.9
gradient_limit = 5.0


sentences_placeholder = tf.placeholder(tf.int32, shape=[None, max_doc_len], name='sentences')
labels_placeholder = tf.placeholder(tf.int32, shape=[None,max_doc_len], name='labels')
sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=[None], name='lengths')
dropout_prob_placeholder = tf.placeholder(tf.float32, name="dropout_prob")

with tf.device("/device:gpu:0"), tf.variable_scope('word-embedding-layer'):
    W_embedding = tf.Variable(name='word-embedding', initial_value=word_lookup_table, dtype=tf.float32, trainable=False)
    vectors = tf.nn.embedding_lookup(W_embedding, sentences_placeholder)

with tf.device("/device:gpu:0"), tf.variable_scope("bi-lstm"):
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                cell_fw, cell_bw, vectors, \
                                sequence_length=sequence_lengths_placeholder, dtype=tf.float32)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.nn.dropout(output, dropout_prob_placeholder)

with tf.device("/device:gpu:0"), tf.variable_scope("projection"):
    r_plus_c = 2*hidden_size_lstm + num_classes
    W = tf.get_variable("W", dtype=tf.float32, \
                        initializer=tf.constant_initializer(np.random.uniform(-sqrt(6.0/r_plus_c), sqrt(6.0/r_plus_c))),\
                        shape=[2*hidden_size_lstm, num_classes])

    b = tf.get_variable("b", shape=[num_classes],dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    output = tf.reshape(output, [-1, 2*hidden_size_lstm])
    pred = tf.matmul(output, W) + b
   
    logits = tf.reshape(pred, [-1, max_doc_len, num_classes])

with tf.device("/device:gpu:0"), tf.name_scope('crf_encode'):

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, labels_placeholder, sequence_lengths_placeholder)
    loss = tf.reduce_mean(-log_likelihood)
    
with tf.device("/device:gpu:0"), tf.name_scope('crf_decode'):
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, trans_params, sequence_lengths_placeholder)
with tf.device("/device:gpu:0"), tf.name_scope('optimizer'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    train_op = optimizer.minimize(loss)



config = tf.ConfigProto(allow_soft_placement = True)

with tf.Session(config = config) as sess:
    sess.run( tf.global_variables_initializer())

    print("Training: Start")

    step = 0
    batches = batch_iter(train_sentences, train_labels, train_sequence_lengths, batch_size=batch_size, num_epochs=n_epochs, shuffle=True)
    timer = time()
    for batch in batches: 
        sent_batch, label_batch, sequence_length_batch = batch

        loss_,_, predicts= sess.run([loss, train_op, viterbi_sequence], feed_dict={
                                                                                sentences_placeholder: sent_batch, 
                                                                                labels_placeholder: label_batch, 
                                                                                sequence_lengths_placeholder: sequence_length_batch,
                                                                                dropout_prob_placeholder: dropout_prob
                                                                            })
        step += 1
        if step % n_batches == 0 or step >= n_batches*n_epochs - 1:
            print("Step %d/%d\tLoss: %f" % (step, n_batches*n_epochs, loss_), end='\t')
            print('Took %fs' % (time() - timer))
            timer = time()

    print()

    del train_sentences, train_labels, train_sequence_lengths
    print("Training: Done")
    
    print("\n\n\n")
    
    print('Developing: Start')
    if os.path.isdir('eval_dev'):
        os.mkdir('eval_dev')
    tsvfile = open('../eval_dev/baseline_predict_file_%s_%s.tsv' % (type_embeddings, tagging), 'w')
    for idx in range(dev_sents.shape[0]):
        sent = dev_sents[idx].reshape(1, -1)
        label = dev_labels[idx].reshape(1, -1)
        sequence_length = np.array([dev_sequence_lengths[idx]])

        predict = sess.run(viterbi_sequence, feed_dict={sentences_placeholder: sent,
                                                        labels_placeholder: label,
                                                        sequence_lengths_placeholder: sequence_length,
                                                        dropout_prob_placeholder: 1.0
                                                        })

        sent = sent[0]
        label = label[0]
        if sequence_length[0] > 0:
            for subidx in range(sequence_length[0]):        
                word = get_word_from_idx(map_id_word, sent[subidx])
                golden_tag = labels_template[label[subidx]]
                predict_tag = labels_template[predict[0][subidx]]
                tsvfile.write("%s\t%s\t%s\n" % (word, golden_tag, predict_tag))
            tsvfile.write('-\tX\t-\n')
        
    print("Developing: Done")
