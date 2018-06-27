

import python_speech_features as p
import scipy.io.wavfile as wav
import glob
import numpy as np
from os.path import join
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import os


def make_mfcc_shape(filename, padlen=778):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}



def get_mfcc(wav_file, max_mfcc_len):
    
    y, sr = librosa.load(wav_file, mono=True) # sr=22050,
    mfcc= librosa.feature.mfcc(y,sr)
    if max_mfcc_len > mfcc.shape[1]:
        mfcc_feature = np.pad(mfcc, ((0, 0), (0, max_mfcc_len-mfcc.shape[1])), 'constant')
    else:
        mfcc_feature = mfcc[:,:max_mfcc_len]
    return mfcc_feature


def create_mfcc_mat(wav_files, path='', save_name='mfcc_vec_680x26', max_mfcc_len=680):
    
    mfcc_mat = []
    for wav_file in tqdm(wav_files):
        mfcc_vec = make_mfcc_shape(wav_file, padlen=max_mfcc_len)
        mfcc_mat.append(mfcc_vec)
    mfcc_mat = np.array(mfcc_mat) # .transpose(0,2,1)
    np.save(join(path, save_name), mfcc_mat)
        
def get_mfcc_mat(path='', save_name='mfcc_vec_680x26'):

    mfcc_mat = np.load(join(path,save_name+'.npy'))
    
    return mfcc_mat






if __name__ == '__main__':    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    path_base = '/data1/1806_speech-recognition/1806_data_speech-recognition/data_thchs30'
    path_data = join(path_base, 'data')
    
    if not os.path.exists(join(path_base,'mfcc_vec_680x26'+'.npy')):
        wav_files = glob.glob(join(path_data ,'*.wav'))
        wav_files.sort()
        print('num of wav files', len(wav_files),'ready to create mfcc mat')
        create_mfcc_mat(wav_files[:],path=path_base) # 第一次创建使用
    else:
        mfcc_mat = get_mfcc_mat(path=path_base)
        print('load from npy',mfcc_mat.shape)