import librosa, glob
from os.path import join
import numpy as np
import codecs
import os
from tqdm import tqdm
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import argparse, platform
from sklearn.metrics import accuracy_score
import keras
from keras.utils import to_categorical
import platform
import itertools
import codecs
import re
import datetime
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam
import keras.callbacks
from keras.utils import multi_gpu_model
from pypinyin import pinyin

from keras.layers.convolutional import Conv1D
from keras.layers import BatchNormalization, Multiply, Add

import pprint
pp = pprint.PrettyPrinter(indent=1)

np.random.seed(2018)

def get_mfcc(wav_file, max_mfcc_len):
    
    y, sr = librosa.load(wav_file, mono=True) # sr=22050,
    mfcc= librosa.feature.mfcc(y,sr)
    if max_mfcc_len > mfcc.shape[1]:
        mfcc_feature = np.pad(mfcc, ((0, 0), (0, max_mfcc_len-mfcc.shape[1])), 'constant')
    else:
        mfcc_feature = mfcc[:,:max_mfcc_len]
    return mfcc_feature


def create_mfcc_mat(wav_files, path='', save_name='mfcc_vec_all', max_mfcc_len=640):
    
    mfcc_mat = []
    for wav_file in tqdm(wav_files):
        mfcc_vec = get_mfcc(wav_file, max_mfcc_len)
        mfcc_mat.append(mfcc_vec)
    mfcc_mat = np.array(mfcc_mat).transpose(0,2,1)
    np.save(join(path, save_name), mfcc_mat)
        
def get_mfcc_mat(path='', save_name='mfcc_vec_all'):

    mfcc_mat = np.load(join(path,save_name+'.npy'))
    
    return mfcc_mat
    
def get_text(text_files):
    lines = []
    for text_file in tqdm(text_files):
        with codecs.open(text_file,encoding='utf-8') as f_read:
            line = f_read.readline()
            lines.append(line.strip().replace(" ", ""))
    return lines

def get_pad_seq(textlines, maxlen=48):

    # saving
    tok_path = join(path_base,'tokenizer_pinyin.pickle')
    pinyin_lines = [pinyin(line) for line in text_lines]
    pinyin_lines = list(map(lambda x:[char[0] for char in x],pinyin_lines))
    if not os.path.exists(tok_path):
        tok=Tokenizer(char_level=True)
        
#         pinyin_lines[:5]
#         tok = Tokenizer()
        tok.fit_on_texts(pinyin_lines)
#         tok.fit_on_texts(text_lines)
        with open(tok_path, 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('create tok')
    # loading
    else:
        with open(tok_path, 'rb') as handle:
            tok = pickle.load(handle)
            print('load tok')
    
#     seq_lines =  tok.texts_to_sequences(text_lines[:])
    seq_lines =  tok.texts_to_sequences(pinyin_lines[:])
    print('num of words,', len(tok.word_index.keys()))
#     print(text_lines[1])
#     print('/'.join(map(lambda x: str(x),seq_lines[1])))
#     print(len(text_lines[1]))

    len_lines = pd.Series(map(lambda x: len(x), seq_lines))
    print('max_len',len_lines.max())
#     len_lines.value_counts()[:5]

    def new_pad_seq(line, maxlen):
        return pad_sequences(line, maxlen=maxlen, padding='post', truncating='post')
    
    lines = seq_lines[:]
    pad_lines = new_pad_seq(lines, maxlen)
#     pad_lines.shape
    return pad_lines, tok, # np.array(len_lines)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model2(img_w=32, img_h=20, output_size=None, max_pred_len=4):


    input_tensor=Input(shape=(img_w,img_h),name='the_input')
    x=Conv1D(kernel_size=1,filters=192,padding="same")(input_tensor)
    x=BatchNormalization(axis=-1)(x)
    x=Activation("tanh")(x)
    
    def res_block(x,size,rate,dim=192):
        x_tanh=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x)
        x_tanh=BatchNormalization(axis=-1)(x_tanh)
        x_tanh=Activation("tanh")(x_tanh)
        x_sigmoid=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x)
        x_sigmoid=BatchNormalization(axis=-1)(x_sigmoid)
        x_sigmoid=Activation("sigmoid")(x_sigmoid)
        out=Multiply()([x_tanh,x_sigmoid]) # ,mode="mul")
        out=Conv1D(kernel_size=1,filters=dim,padding="same")(out)
        out=BatchNormalization(axis=-1)(out)
        out=Activation("tanh")(out)
        x=Add()([x,out]) # ,mode="sum")
        return x,out

    skip=[]
    for i in np.arange(0,3):
        for r in [1,2,4,8,16]:
            x,s=res_block(x,size=7,rate=r)
            skip.append(s)


    skip_tensor=Add()([s for s in skip]) # ,mode="sum")
    logit=Conv1D(kernel_size=1,filters=192,padding="same")(skip_tensor)
    logit=BatchNormalization(axis=-1)(logit)
    logit=Activation("tanh")(logit)
    y_pred=Conv1D(kernel_size=1,filters=output_size,padding="same",activation="softmax")(logit)

    # Model(inputs=input_tensor, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[max_pred_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    opt = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # opt = Adam(lr=0.001)
    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    test_func = K.function([input_tensor,K.learning_phase()], [y_pred])
    if os.path.exists(join(path_base,"best_weights_680x26_pinyin.h5")) :
        
        model.load_weights(join(path_base,"best_weights_680x26_pinyin.h5"))
        print('load weights from', join(path_base,"best_weights_680x26_pinyin.h5"))
    
    return model, test_func


def get_batch(x, y, train=False, max_pred_len=4, input_length=8):
    
    
    X = np.expand_dims(x, axis=3)
    X = x # for model2
#     labels = np.ones((y.shape[0], max_pred_len)) *  -1 # 3 # , dtype=np.uint8
    labels = y
    
    input_length = np.ones([x.shape[0], 1]) * ( input_length - 2 )
#     label_length = np.ones([y.shape[0], 1])
    label_length = np.sum(labels > 0, axis=1)
    label_length = np.expand_dims(label_length,1)

    inputs = {'the_input': X,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              }
    outputs = {'ctc': np.zeros([x.shape[0]])}  # dummy data for dummy loss function
    return (inputs, outputs)
    
def decode_batch(test_func, batch):
    out = test_func([batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
#         outstr = labels_to_text(out_best)
        ret.append(out_best)
    return ret

    
class MetricCallback(keras.callbacks.Callback):
    
    def __init__(self, test_func, x, y, idx2w, num_test_words=18,info='this is test'):
        self.test_func = test_func
        self.x = x
        self.y = y
        self.idx2w = idx2w
        self.num_test_words = num_test_words
        self.info = info
    
    def on_epoch_end(self, epoch, logs={}):
        
        
        y_pred = decode_batch(self.test_func, self.x[0:self.num_test_words])
        y_true = self.y[:self.num_test_words]
        y_pred = [''.join(map(lambda x: self.idx2w[x],pred)) for pred in y_pred]
        y_true = [''.join(map(lambda x: self.idx2w[x],true)) for true in y_true]
        
        random_idx = np.random.randint(0, self.num_test_words)
        
        print('\n'+self.info)
        print('pred=',y_pred[random_idx])
        print('true=',y_true[random_idx])
        
        num_shot = sum([len(set(pred) & set(true)) for pred, true in zip(y_true, y_pred)])
        num_true = sum([len(true) for true in y_true])
        print('accuracy:',num_shot / num_true)

if __name__ == '__main__':    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    path_base = '/data1/1806_speech-recognition/1806_data_speech-recognition/data_thchs30'
    path_data = join(path_base, 'data')
    
#     K.set_learning_phase(1) #set learning phase
    
    if not os.path.exists(join(path_base,'mfcc_vec_680x26'+'.npy')):
        wav_files = glob.glob(join(path_data ,'*.wav'))
        wav_files.sort()
        print('num of wav files', len(wav_files),'ready to create mfcc mat')
        create_mfcc_mat(wav_files[:],path=path_base) # 第一次创建使用
    else:
        mfcc_mat = get_mfcc_mat(path=path_base,save_name='mfcc_vec_680x26')
        print('load from npy',mfcc_mat.shape)

    
    text_path = join(path_base, 'all_texts.txt')
    if not os.path.exists(text_path):
        text_files = glob.glob(join(path_data ,'*.wav.trn'))
        text_files.sort()
        print('num of trn files', len(text_files), 'ready to create text file')
        text_lines = get_text(text_files[:])
        with codecs.open(text_path,mode='w',encoding='utf-8') as f_write:
            for line in text_lines:
                f_write.write(line + '\n')
    else:
        text_lines = []
        with codecs.open(text_path,encoding='utf-8') as f_read:
            lines = f_read.readlines()
            for line in lines:
                text_lines.append(line.strip().replace(" ", ""))
        print('load from text file',len(text_lines))
    
    
    pad_lines, tok = get_pad_seq(text_lines, maxlen=48)
    pad_lines.shape
    

    x_train, x_test, y_train, y_test = train_test_split(mfcc_mat, pad_lines, test_size=0.2)
    mmx = MinMaxScaler()
    train_shape = x_train.shape
    x_train = mmx.fit_transform(x_train.reshape(-1,1)).reshape(train_shape)
    test_shape = x_test.shape
    x_test = mmx.transform(x_test.reshape(-1,1)).reshape(test_shape)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
    model, test_func = get_model2(img_w=680, img_h=26, output_size=1073 + 2, max_pred_len=48)
    x_train2, y_train2 = get_batch(x_train[:10640], y_train[:10640], max_pred_len=48, input_length=680)

    x_test2, y_test2 = get_batch(x_test[:640], y_test[:640], max_pred_len=48, input_length=680)
    
    idx2w = dict((i,w) for w,i in tok.word_index.items())
    max_key = max(idx2w.keys())
    idx2w[0] = ''
    idx2w[max_key + 1] = ''
    metric_cb_test = MetricCallback(test_func, x_test[:200], y_test[:200], idx2w,num_test_words=200,info='this is test')
    metric_cb_train = MetricCallback(test_func, x_train[:200], y_train[:200], idx2w,num_test_words=200,info='this is train')
    checkpointer = keras.callbacks.ModelCheckpoint(join(path_base,"best_weights_680x26_pinyin.h5"), verbose=1, save_best_only=False, save_weights_only=True, period=3)
    lr_change = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_lr=0.000,epsilon=0.01,verbose=1)
    csv_to_log = CSVLogger(join(path_base, "logger_0622.csv"))
    model.fit(x=x_train2, y=y_train2, batch_size=80, epochs=1000, validation_data=(x_test2, y_test2),initial_epoch =0,callbacks=[csv_to_log,metric_cb_test,metric_cb_train,checkpointer,lr_change])