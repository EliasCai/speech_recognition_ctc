# coding: utf-8

# 使用清华开源的中文识别语料进行训练

from os.path import join
import numpy as np
import os
import sys
import itertools
import shutil

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import keras
import keras.backend as K
from keras.callbacks import CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Activation, Lambda
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model
from keras.layers import BatchNormalization, Multiply, Add
from utils import get_wav_files, get_corpus, remove_blank_wav, get_batch


np.random.seed(2018)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_w=32, img_h=20, output_size=None, max_pred_len=4,model_path='best.h5'):


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
        out=Multiply()([x_tanh,x_sigmoid]) 
        out=Conv1D(kernel_size=1,filters=dim,padding="same")(out)
        out=BatchNormalization(axis=-1)(out)
        out=Activation("tanh")(out)
        x=Add()([x,out]) 
        return x,out

    skip=[]
    for i in np.arange(0,3):
        for r in [1,2,4,8,16]:
            x,s=res_block(x,size=7,rate=r)
            skip.append(s)


    skip_tensor=Add()([s for s in skip]) 
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
#    model = multi_gpu_model(model, gpus=2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    test_func = K.function([input_tensor,K.learning_phase()], [y_pred])
    if os.path.exists(model_path) :
        model.load_weights(model_path)
        print('load weights from', model_path)
    else:
        print(model_path,'do not exists!')
    
    return model, test_func


  
def decode_batch(test_func, batch):
    out = test_func([batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
#         outstr = labels_to_text(out_best)
        ret.append(out_best)
    return ret

    
def get_mms(path, mms_batch):
    
    mms_path = join(path,'mms.pkl')
    if os.path.exists(mms_path):
        mms = joblib.load(mms_path) 
        print('loading mms')
    else:
        mms = MinMaxScaler()
        x, _ = next(mms_batch)
        mms.fit(x['the_input'].reshape((-1.1)))
        joblib.dump(mms, mms_path)
        print('training mms')
        
    return mms
    
    
    
    

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
    
    path = 'data_aishell'
    
    K.set_learning_phase(1) #set learning phase

    os.environ["CUDA_VISIBLE_DEVICES"] = "1" #　选择使用的GPU
    
    train_wavs, val_wavs, test_wavs = get_wav_files(path)
    name_to_seq, tok = get_corpus(path,maxlen=48)
    
    train_wavs = remove_blank_wav(train_wavs, name_to_seq)     
    val_wavs = remove_blank_wav(val_wavs, name_to_seq) 
    test_wavs = remove_blank_wav(test_wavs, name_to_seq)   
    
    model_path = join(path,"best_weights_778x26.h5")
    log_path = join(path,'logs')
    
    model, test_func = get_model(img_w=778, img_h=26, 
                                 output_size=len(tok.word_index) + 2, max_pred_len=48,
                                 model_path=model_path)
    
    print(len(train_wavs))
    print(len(name_to_seq))
    
    
    mms_batch = get_batch(train_wavs[:], name_to_seq,
                          batch_size=1000, max_pred_len=48, input_length=778, 
                          mms=None)
    
    mms = get_mms(path, mms_batch)
    
    train_batch = get_batch(train_wavs[:2000], name_to_seq,
                            batch_size=64, max_pred_len=48, input_length=778, 
                            mms=mms)
    
    val_batch = get_batch(val_wavs, name_to_seq,
                          batch_size=16, max_pred_len=48, input_length=778, 
                          mms=mms)
    
    ctc_x, ctc_y = next(train_batch)
    
    for name, value in ctc_x.items():
        print(name, value.shape)
        
    for name, value in ctc_y.items():
        print(name, value.shape)  
    
    checkpointer = ModelCheckpoint(model_path, verbose=1, 
                                   save_best_only=False, 
                                   save_weights_only=True, period=1)
    csv_to_log = CSVLogger(join(path, "logger_0627.csv"))
    lr_change = ReduceLROnPlateau(monitor="loss", factor=0.5, 
                                  patience=1, min_lr=0.000,
                                  epsilon=0.1,verbose=1)
    tfboard = TensorBoard(log_dir=log_path)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    
    callback_list = [checkpointer, tfboard, lr_change]
    history = model.fit_generator(train_batch,
                                  steps_per_epoch=40, epochs=100,
                                  callbacks=callback_list,
                                  workers=2, max_queue_size=256)    

