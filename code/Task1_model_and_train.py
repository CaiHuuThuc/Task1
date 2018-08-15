import os
import shelve
import numpy as np
import tensorflow as tf
import time
import csv
from Task1_datahelper import preprocess


data = preprocess()
max_doc_len = data['max_doc_len']
labels_template = data['labels_template']
train_sent = data["train_sentences"]

train_vectors = data["train_vectors"]

train_labels = data["train_labels"]

train_sequence_lengths = data["train_sequence_lengths"]


dev_sent = data["dev_sentences"]
dev_vectors = data["dev_vectors"]
dev_labels = data["dev_labels"]
dev_sequence_lengths = data["dev_sequence_lengths"]


labels_template = np.array(labels_template)
train_sent = np.array(train_sent)
train_vectors = np.array(train_vectors)
train_labels = np.array(train_labels)
train_sequence_lengths = np.array(train_sequence_lengths)


embedding_size = 200
num_classes = len(labels_template)
hidden_size_lstm = 120
dropout_prob = 0.5
n_epochs = 1
batch_size = 32
n_batches = int(train_vectors.shape[0]//batch_size) + 1
learning_rate = 0.001
momentum = 0.9

vectors_placeholder = tf.placeholder(tf.float32, shape=[None, max_doc_len, embedding_size])
labels_placeholder = tf.placeholder(tf.int32, shape=[None,max_doc_len])
sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=[None])

train_dataset = tf.data.Dataset \
                .from_tensor_slices((vectors_placeholder, labels_placeholder, sequence_lengths_placeholder)) \
                .shuffle(buffer_size=2).batch(batch_size).repeat()


train_iterator = train_dataset.make_initializable_iterator()


x, y, sequence_length = train_iterator.get_next()

with tf.variable_scope("bi-lstm"):
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                cell_fw, cell_bw, x, \
                                sequence_length=sequence_length, dtype=tf.float32)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.nn.dropout(output, dropout_prob)

with tf.variable_scope("projection"):
    W = tf.get_variable("W", dtype=tf.float32, shape=[2*hidden_size_lstm, num_classes])

    b = tf.get_variable("b", shape=[num_classes],dtype=tf.float32, initializer=tf.zeros_initializer())

    output = tf.reshape(output, [-1, 2*hidden_size_lstm])
    pred = tf.matmul(output, W) + b
   
    logits = tf.reshape(pred, [-1, max_doc_len, num_classes])

with tf.name_scope('crf_encode'):

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, y, sequence_length)
    loss = tf.reduce_mean(-log_likelihood)
with tf.name_scope('crf_decode'):
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, trans_params, sequence_length)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    sess.run(train_iterator.initializer, feed_dict={vectors_placeholder:train_vectors,
                                            labels_placeholder:train_labels,
                                            sequence_lengths_placeholder:train_sequence_lengths})
    step = 0
    print("Training: Start")
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            loss_, _, _, _, _ = sess.run([loss, optimizer, x, y, sequence_length])
            step += 1
            if step % 10 == 0:
                print("Epoch %d/%d  Batch %d/%d Loss: %f" % (epoch, n_epochs - 1, batch, n_batches - 1, loss_))
        print()

    del train_sent, train_vectors, train_labels, train_sequence_lengths
    print("Training: Done")
    
    print("\n\n\n")
    
    print('Developing: Start')
    ids = np.array(list(range(len(dev_sent))))
    idx_placeholder = tf.placeholder(tf.int32, shape = [None])
    dev_dataset = tf.data.Dataset \
                .from_tensor_slices((idx_placeholder, vectors_placeholder, labels_placeholder, sequence_lengths_placeholder)).batch(1)
    dev_iterator = dev_dataset.make_initializable_iterator()
    

    dev_sent = np.array(dev_sent)
    dev_vectors = np.array(dev_vectors)
    dev_labels = np.array(dev_labels)
    dev_sequence_lengths = np.array(dev_sequence_lengths)
    predict_tags = []
    dev_idx, dev_x, dev_y, dev_sequence_length = dev_iterator.get_next()
    sess.run(dev_iterator.initializer, feed_dict={
                                            idx_placeholder: ids,
                                            vectors_placeholder:dev_vectors,
                                            labels_placeholder:dev_labels,
                                            sequence_lengths_placeholder:dev_sequence_lengths})

    f = open('predict_file.tsv', 'w')
    tsvfile = csv.writer(f, delimiter=' ')
        
    for _ in range(dev_vectors.shape[0]):
        predict = sess.run(viterbi_sequence)
        idx_sent, _, ground_labels, seq_len = sess.run([dev_idx, dev_x, dev_y, dev_sequence_length])
        print(predict.shape)
        

        for subidx in range(len(seq_len)):
            word = dev_sent[idx_sent,subidx]
            golden_tag = labels_template[ground_labels[subidx]]
                
            predict_tag = labels_template[predict[subidx]]
            tsvfile.writerow([word, golden_tag, predict_tag])
        tsvfile.writerow('-X-')
        
    f.close()
    print("Developing: Done")

    print("\n")


    print("Writing predict to file")

    print("Done")
