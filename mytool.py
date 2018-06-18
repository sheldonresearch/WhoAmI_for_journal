#!/usr/bin/env python
# encoding: utf-8


"""
@version: python3.6
@author: Xiangguo Sun
@contact: sunxiangguodut@qq.com
@site: http://blog.csdn.net/github_36326955
@software: PyCharm
@file: mytool
@time: 17-8-15 1:35pm
"""
import os
import sys
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from config import GLOVE_DIR

def get_embeddings():
    embeddings_index = {}
    with open(GLOVE_DIR) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index



def load_data_labels(TEXT_DATA_DIR):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath)
                        #f = open(fpath,mode='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)
    return texts, labels, labels_index

def log_results(path, accuracy,macro_precision,recall,f1,precision):
    with open(path, "a") as f:
        f.write("accuracy score for classification model - ")
        f.write(str(accuracy))
        f.write("\n\n")

        f.write("macro Precision score for classification model - ")
        f.write(str(macro_precision))
        f.write("\n\n")

        f.write("recall score for classification model - ")
        f.write(str(recall))
        f.write("\n\n")

        f.write("f1 score for classification model - ")
        f.write(str(f1))
        f.write("\n\n")

        f.write("precision for classification model - ")
        f.write(str(precision))
        f.write("\n\n")


#acc, macro_precision, recall, f1,precision

def get_result(y_test, y_pred):
    accuracy=accuracy_score(y_test, y_pred)
    print("accuracy for classification model - ", accuracy)

    macro_precision = precision_score(y_test, y_pred, average='weighted')
    print("macro Precision score for classification model - ", macro_precision)

    recall = recall_score(y_test, y_pred,)
    print("recall score for classification model - ", recall)

    f1=f1_score(y_test, y_pred)
    print("f1 score for classification model - ", f1)

    precision = precision_score(y_test, y_pred)
    print("Precision score for classification model - ", precision)

    return (accuracy,macro_precision,recall,f1,precision)
