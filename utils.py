# -*- coding: utf-8 -*-


import glob
from os.path import join
import codecs
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import python_speech_features as p
import scipy.io.wavfile as wav
import numpy as np
import pickle

np.random.seed(2018)

def get_wav_paths(path, data_type='train'):
    
    find_path = join(path,'wav', data_type,'*','*.wav')
    wav_files = glob.glob(find_path)
    
    return wav_files

def get_trans_text(path):
    
    text_trans = join(path, 'transcript', 'aishell_transcript_v0.8.txt')
    with codecs.open(text_trans, encoding='utf-8') as f:
        lines = f.readlines()
    
    return lines


def get_name_to_text(lines):
    
    name_to_text = {}
    for line in lines:
        name, text = line.split(' ',1)
        name_to_text[name] = text.strip().replace(' ','') 
    
    return name_to_text

def split_train_val(wav_paths, test_size=0.2):
    
    return train_test_split(wav_paths, test_size=test_size)


def get_token(name_to_text, path):
    
    
    tok_path = join(path,'tokenizer.pickle')
    # saving
    if not os.path.exists(tok_path):
        tok = Tokenizer(char_level=True)
        texts = list(name_to_text.values())
        tok.fit_on_texts(texts)
        with open(tok_path, 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('create tok')
    # loading
    else:
        with open(tok_path, 'rb') as handle:
            tok = pickle.load(handle)
            print('load tok')
    
    return tok


def get_name_to_seq(name_to_text, tok, maxlen=48):
    
    name_to_seq = {}
    for name, text in name_to_text.items():
        seq = tok.texts_to_sequences([text])
        pad_seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
        pad_seq = pad_seq[0]
        name_to_seq[name] = pad_seq
        
    return name_to_seq


def get_wav_files(path,val_size=0.2):
    
    train_wavs = get_wav_paths(path, 'train')
    train_wavs, val_wavs = split_train_val(train_wavs,test_size=val_size)
    test_wavs = get_wav_paths(path, 'test')
    
    return train_wavs, val_wavs, test_wavs


def get_corpus(path,maxlen=48):

    trans_lines = get_trans_text(path)
    name_to_text = get_name_to_text(trans_lines)
    tok = get_token(name_to_text, path)
    name_to_seq = get_name_to_seq(name_to_text, tok, maxlen=48)

    return name_to_seq, tok


def make_mfcc_shape(filename, padlen=778):
    
    fs, audio = wav.read(filename)
    # 2D array -> timesamples x mfcc_features
    r = p.mfcc(audio, samplerate=fs, numcep=26)  
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', 
                      padding='post', truncating='post').T
    
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}


def remove_blank_wav(wav_files, name_to_seq):
    
    remove_list = []
#    wav_names = [os.path.basename(file).split('.')[0] for file in wav_files]
#    text_name = list(name_to_seq.keys())
#    blank_wav = list(set(wav_names)- set(text_name))
    for file in wav_files:
        name = os.path.basename(file).split('.')[0]
        if name not in name_to_seq:
            remove_list.append(file)
#    print(len(remove_list))
    for file in remove_list:
        wav_files.remove(file)
    
    return wav_files

def feed_ctc(x, y, max_pred_len=48, input_length=778):
     
    X = x 
    labels = y
    
    input_length = np.ones([x.shape[0], 1]) * ( input_length - 2 )
    label_length = np.sum(labels > 0, axis=1)
    label_length = np.expand_dims(label_length,1)

    inputs = {'the_input': X,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              }
    outputs = {'ctc': np.zeros([x.shape[0]])}  # dummy data for dummy loss function
    return (inputs, outputs)

def get_batch(wav_files, name_to_seq, batch_size=16, max_pred_len=48, input_length=778):
    
    while True:
        batch_files = np.random.choice(wav_files,batch_size)
        batch_mfcc = []
        batch_seq = []
        for file in batch_files:
            name = os.path.basename(file).split('.')[0]
            feat = make_mfcc_shape(file, padlen=input_length)
            batch_mfcc.append(feat)
            batch_seq.append(name_to_seq[name])
            
        x = np.array(batch_mfcc)
        y = np.array(batch_seq)
        yield feed_ctc(x, y, max_pred_len=max_pred_len, input_length=input_length)

if __name__ == '__main__':
    
    
    path = 'data_aishell'
    
    train_wavs = get_wav_paths(path, 'train')
    
    train_wavs, val_wavs = split_train_val(train_wavs,test_size=0.2)
    
    test_wavs = get_wav_paths(path, 'test')
    
    trans_lines = get_trans_text(path)
    
    name_to_text = get_name_to_text(trans_lines)
    
    tok = get_token(name_to_text)
    
    all_wavs = train_wavs + test_wavs
    wav_names = [os.path.basename(wav).split('.')[0] for wav in all_wavs]
    
    drop_keys = set(name_to_text.keys()) - set(wav_names)
    
    
    print(len(tok.word_index))
#    print('train set =',len(train_wavs),'val set =',len(val_wavs))
#    print(train_wavs[:5])
#    print(os.path.basename(train_wavs[4]))
#    print(test_wavs[:5])
#    
#    print(wav_names[:5])
#    
#    print(len(name_to_text))
#    
#    print(name_to_text['BAC009S0002W0126'])
#    print(len(drop_keys))
#    
#    print(list(name_to_text.values())[:5])
#    
#    print('max len =',max([len(value) for value in list(name_to_text.values())]))
#    
#    for key in list(drop_keys)[:5]:
#        
#        print(name_to_text[key])
    
         