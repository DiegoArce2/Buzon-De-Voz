#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue
import argparse
import time
from scipy.io import wavfile
from scipy import signal
start = time.time()

def extract_feature(file_name=None):
    if file_name: 
        print('Extracting', file_name,' Tiempo actual: ',(time.time()-start))
        preprocesado(file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:  
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True: 
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    return mfccs,mel

def preprocesado(file_name):
        
        try:
            sr, x = wavfile.read(file_name)      # 16-bit mono 44.1 khz

            b = signal.firwin(5, cutoff=1000, fs=sr, pass_zero=False)

            x = signal.lfilter(b, [1.0], x)

            wavfile.write(file_name, sr, x.astype(np.int16))
            
            print("pre-procesado ejecutado correctamente")
        except Exception as e:
            print("Error de pre-procesado"+e)


def parse_audio_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,168)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try: mfccs,mel= extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn,e))
                    continue
                ext_features = np.hstack([mfccs,mel])
               
                features = np.vstack([features,ext_features])
                # labels = np.append(labels, fn.split('/')[1])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def parse_predict_files(parent_dir,file_ext='*.wav'):
    features = np.empty((0,168))
    filenames = []
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        mfccs,mel= extract_feature(fn)
        ext_features = np.hstack([mfccs,mel])
        features = np.vstack([features,ext_features])
        filenames.append(fn)
        print("extract %s features done" % fn)
    return np.array(features), np.array(filenames)

def main(args):
    if args.all:
        print("Analizando todo")
        features, labels = parse_audio_files('data')
        np.save('feat.npy', features)
        np.save('label.npy', labels)

        # Predict new
        features, filenames = parse_predict_files('predict')
        np.save('predict_feat.npy', features)
        np.save('predict_filenames.npy', filenames)
    if(args.pr):
        print("Analizando predict")
        features, filenames = parse_predict_files('predict')
        np.save('predict_feat.npy', features)
        np.save('predict_filenames.npy', filenames)

    # Get features and labels
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-all', '--all',action='store_true',help='Entrenar todo')
    parser.add_argument('-pr', '--pr',action='store_true',help='Analizar')

    args = parser.parse_args()
    main(args)

