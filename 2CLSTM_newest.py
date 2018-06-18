#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: Xiangguo Sun
@contact: sunxiangguo@seu.edu.cn
@site: http://blog.csdn.net/github_36326955
@software: PyCharm
@file: 2CLSTM.py
@time: 17-7-27 5:15pm
"""
import numpy as np

from keras import metrics
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import MaxPooling3D,BatchNormalization,Dense,Reshape, Flatten,LSTM,Conv2D,Bidirectional, Concatenate,Dropout, Input, TimeDistributed,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from mytool import load_data_labels,get_embeddings

from config import class_type,TRAIN_DATA_DIR,TEST_DATA_DIR, bs

from keras import regularizers
#
#
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# #进行配置，使用70%的GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = tf.Session(config=config)
#
# # 设置session
# KTF.set_session(session )


print("pre_handle data...")


# parameters section
input_length = 500 # assume each sample consists of input_length words.
MAX_NB_WORDS = 10000
w2vDimension = 100
hidden_dim_1 = 2 * w2vDimension
'''
hidden_dim_1 = 300
this is dimention for left context and right context respectively.
I donnot think set hidden_dim_1 = 300 is reasonable, because word vector is only 100 (w2vDimension)
2 times of that is enough becuase we have bidirectional LSTM, which will be up to 4 times!
'''


hidden_dim_2 = 300
'''
hidden_dim_2 = 300
this is dimention for "fully connected layer" in Figure: The architecture of 2CLSTM.
the input shape for this layer is 2* hidden_dim_1+w2vDimension=500
'''



# end parameters section



embeddings_index = get_embeddings()
texts, labels, labels_index = load_data_labels(TRAIN_DATA_DIR)
texts_test, labels_test, labels_indeX_test = load_data_labels(TEST_DATA_DIR)
tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences_test = tokenizer.texts_to_sequences(texts_test)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=input_length)
data_test = pad_sequences(sequences_test, maxlen=input_length)
labels_cat = to_categorical(np.asarray(labels))
# labels_cat=np.asarray(labels)
# print(np.shape(labels))
# print(np.shape(labels_cat))


print("shuffle data and labels_cat...")


index = np.arange(data.shape[0])
np.random.seed(1024)
np.random.shuffle(index)
data=data[index]
labels_cat=labels_cat[index]

print("shuffle done!")



X_train = data
y_train_cat = labels_cat
y_train = np.asarray(y_train_cat.argmax(axis=1))
# y_train=y_train_cat

X_test = data_test
y_test = np.asarray(labels_test)
y_test_cat = to_categorical(y_test)

classes = len(labels_index)

n_symbols = min(MAX_NB_WORDS, len(word_index))+1
embedding_weights =0.5*np.random.random((n_symbols,w2vDimension))-0.25  # np.zeros((n_symbols, w2vDimension))

# embedding_weights = np.random((n_symbols, w2vDimension))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_weights[i] = embedding_vector


#start our model

input_a = Input(shape=(input_length,))

embedding_layer=Embedding(output_dim=w2vDimension,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length,
                        trainable=False)(input_a)

embedding_bn=BatchNormalization(axis=1)(embedding_layer)

blstm = Bidirectional(LSTM(hidden_dim_1,
                        bias_initializer='truncated_normal', return_sequences=True))(embedding_bn)

concatenate = Concatenate(axis=2)([blstm,embedding_layer])

bn_layer=BatchNormalization(axis=1)(concatenate)  # 500*700

reshape=Reshape(target_shape=(input_length,2*hidden_dim_1+w2vDimension,1))(bn_layer)



sentence_length=5 # assume each sentence consists of sentence_length words
column=2*hidden_dim_1+w2vDimension-hidden_dim_2+1
sentence_vectors=Dropout(rate=0.2)(Conv2D(filters=5,
                kernel_size=(sentence_length,column),
                strides=(sentence_length, 1),
                        bias_initializer='truncated_normal',
                # kernel_regularizer=regularizers.l1_l2(0.01,0.01),
                # activity_regularizer=regularizers.l1_l2(0.01,0.01),
                activation="relu")(reshape))
'''
if strides=(1,1)
shape: (None, 491,300,10)
shape: (None, input_length-sentence_length+1,hidden_dim_2,filters=10)

if strides=(sentence_length,1)
shape: (None, 50,300,10)
shape: (None, int(input_length/sentence_length),hidden_dim_2,filters=10)
'''

d1,d2,d3=int(sentence_vectors.shape[1]),\
            int(sentence_vectors.shape[2]),\
            int(sentence_vectors.shape[3])

# sentence_vectors_reshape=Reshape(target_shape=(d1,d2,d3,1))(sentence_vectors)
# pool_rnn=MaxPooling3D(pool_size=(1,1,10))(sentence_vectors_reshape)

sentence_vectors_reshape=Reshape(target_shape=(d1,d2,d3))(sentence_vectors)
pool_rnn=sentence_vectors
# pool_rnn=Reshape(target_shape=(int(pool_rnn.shape[1]),
#                                int(pool_rnn.shape[2]),
#                                1))(pool_rnn)






conv2=Dropout(rate=0.2)(Conv2D(filters=5,
             kernel_size=(1,int(pool_rnn.shape[2])),
             # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
             # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
             activation="relu")(pool_rnn))
conv2_reshape=Reshape(target_shape=(int(conv2.shape[1]),int(conv2.shape[3]),int(conv2.shape[2])))(conv2)
conv2_pooling=MaxPooling2D(pool_size=(1,int(conv2_reshape.shape[2])))(conv2_reshape)

conv3=Dropout(rate=0.2)(Conv2D(filters=5,
             kernel_size=(2,int(pool_rnn.shape[2])),
             # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
             # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
             activation="relu")(pool_rnn))
conv3_reshape=Reshape(target_shape=(int(conv3.shape[1]),int(conv3.shape[3]),int(conv3.shape[2])))(conv3)
conv3_pooling=MaxPooling2D(pool_size=(1,int(conv3_reshape.shape[2])))(conv3_reshape)


conv4=Dropout(rate=0.2)(Conv2D(filters=5,
             kernel_size=(3,int(pool_rnn.shape[2])),
             # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
             # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
             activation="relu")(pool_rnn))
conv4_reshape=Reshape(target_shape=(int(conv4.shape[1]),int(conv4.shape[3]),int(conv4.shape[2])))(conv4)
conv4_pooling=MaxPooling2D(pool_size=(1,int(conv4_reshape.shape[2])))(conv4_reshape)


conv2_bn=BatchNormalization(axis=1)(conv2_pooling)
conv3_bn=BatchNormalization(axis=1)(conv3_pooling)
conv4_bn=BatchNormalization(axis=1)(conv4_pooling)

concatenate = Concatenate(axis=1)([conv2_bn,conv3_bn, conv4_bn])
# concatenate=Dropout(rate=0.25)(concatenate)

# concatenate_reshape=Reshape(target_shape=(int(concatenate.shape[1]),int(concatenate.shape[3]),int(concatenate.shape[2])))(concatenate)
# concatenate_pooling=MaxPooling2D(pool_size=(1,int(concatenate_reshape.shape[2])))(concatenate_reshape)


#
# concat_reshape=Reshape(target_shape=(int(concatenate.shape[1]),int(concatenate.shape[3]),int(concatenate.shape[2])))(concatenate)
# conv5=Dropout(rate=0.25)(Conv2D(filters=5,kernel_size=(3,int(concatenate_pooling.shape[2])),
#                         activation="relu")(concatenate_pooling))

conv5_bn=BatchNormalization(axis=1)(concatenate)
flat=Flatten()(conv5_bn)

# flat=Flatten()(concatenate)
dense_layer=BatchNormalization(axis=1)(flat)

# dense_layer=Dense(500,
#                   activation="relu")(dense_layer)


# dense_layer=Flatten()(pool_rnn)

output = Dense(2,
               activation="softmax",
               kernel_regularizer=regularizers.l1_l2(0.02, 0.02),
               activity_regularizer=regularizers.l1_l2(0.02, 0.02),
# kernel_initializer='truncated_normal',
#                         bias_initializer='truncated_normal'
               )(dense_layer)
model = Model(outputs=output,inputs=input_a )
# model end
from keras.models import load_model
try:
    print("load pre_trained weigt for ",class_type,"...")
    model = load_model('./logs/weight_not_augment_'+class_type+'.hdf5')
    print("pre_trained weight loaded!")
except:
    print("OOPs: pre_trained weight loaded failed!!")

#compile the model
"""
'categorical_crossentropy'
'binary_crossentropy'
'mse'
'mae'
'hinge'
'kullback_leibler_divergence'
'cosine_proximity'

"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


#from keras.models import load_model
# try:
#     print("load pre_trained weigt...")
#     model = load_model('./logs/weight_not_augment_'+class_type+'.hdf5')
#     print("pre_trained weight loaded!")
# except:
#     print("OOPs: pre_trained weight loaded failed!!")


# callbacks:
tb = TensorBoard(log_dir='./logs/nonaug',  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=bs,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

es=EarlyStopping(monitor='val_loss', patience=20, verbose=0)

mc=ModelCheckpoint(
    './logs/weight_not_augment_'+class_type+'.hdf5',
    monitor='val_loss', #'val_loss',  'val_categorical_accuracy'
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

rp=ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.9,
    patience=30,
    verbose=0,
    mode='min',
    epsilon=0.001,
    cooldown=0,
    min_lr=0.001
)

# callbacks = [es,tb,rp]
callbacks = [es,tb,mc,rp]


# start to train out model

ne = 1000
hist = model.fit(data, labels_cat,batch_size=bs,epochs=ne,
                      verbose=2,validation_split=0.3,callbacks=callbacks)

print("train process done!!")