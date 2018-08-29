import re
import os
import numpy as np
import tensorflow as tf
def readFileTSV(filename='../eval_dev/predict_file.tsv'):
    sentences = [[]]
    labels = [[]]
    predicts = [[]]
    
   
    with open(filename) as f:
        for _, line in enumerate(f.readlines()):
            row = re.split("\t", line)
            if row[1] == 'X' :
                sentences.append([])
                labels.append([])
                predicts.append([])

            # elif len(row[1]) > 1 and re.split('-', row[1])[0] == 'I':
            #     pass
            else:
                sentences[-1].append(row[0])
                labels[-1].append(row[1])
                predicts[-1].append(row[2].strip())

    return sentences, labels, predicts



def my_eval(sentences, labels, predicts, lengths):
    def get_labels_template(labels):
        labels_template = []
        for label in labels:
            for sublabel in label:
                if sublabel not in labels_template:
                    labels_template.append(sublabel)
        return labels_template
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    TRUE_NEGATIVE = 2
    FALSE_NEGATIVE = 3

    labels_template = get_labels_template(labels)
    eval_matrix = np.zeros(shape=[len(labels_template), 4], dtype=np.int32)
    for idx_label_template, label_template in enumerate(labels_template):

        for idx_sent in range(len(lengths)):
            label = labels[idx_sent]
            predict = predicts[idx_sent]
            for idx_word in range(lengths[idx_sent]):
                ground_truth = label[idx_word]
                predict_tag = predict[idx_word]
                colx = 0
                if ground_truth == label_template: #positive
                    colx = 0
                else: #negative
                    colx = 2
                if ground_truth == predict_tag: #true
                    colx += 0
                else: #false
                    colx += 1
                eval_matrix[idx_label_template, colx] += 1
    TP = eval_matrix[:,TRUE_POSITIVE].sum()
    FP = eval_matrix[:,FALSE_POSITIVE].sum()
    FN = eval_matrix[:,FALSE_NEGATIVE].sum()
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    tf.summary.scalar("Precision", precision)
    tf.summary.scalar("Recall", recall)
    tf.summary.scalar("F1-score", F1)
    return precision, recall, F1
if __name__ == '__main__':
    sentences, labels, predicts = readFileTSV()
    precision, recall, F1 = eval(sentences, labels, predicts)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)