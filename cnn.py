#!/usr/bin/env python
# coding= UTF-8

import feat_extract
from feat_extract import *
import time
import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import os
import os.path as op
from sklearn.model_selection import train_test_split

def train(args):
    if not op.exists('feat.npy') or not op.exists('label.npy'):
        if input('No se encontraron etiquetas extraidas, ¿Desea extraerlas? (Y/n)').lower() in ['y', 'yes', '']:
            feat_extract.main()
            train(args)
    else:
        X = np.load('feat.npy')
        y = np.load('label.npy').ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=233)

    # Count the number of sub-directories in 'data'
    class_count = len(next(os.walk('data/'))[1])

    # Build the Neural Network
    model = Sequential()
    model.add(Conv1D(64, 6, activation='relu', input_shape=(168, 1)))
    model.add(Conv1D(64, 6, activation='relu'))
    model.add(MaxPooling1D(6))
    model.add(Conv1D(128, 6, activation='relu'))
    model.add(Conv1D(128, 6, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Convert label to onehot
    y_train = keras.utils.to_categorical(y_train, num_classes=class_count)
    y_test = keras.utils.to_categorical(y_test, num_classes=class_count)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    start = time.time()
    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)
    print('Puntaje del test:', score)
    print('Exactitud:', acc)
    print('Tiempo:', (time.time()-start))
    model.save("trained_model.h5")



def main(args):
    print(args)
    if args.train: train(args)
    elif args.predict: predict(args)
    elif args.real_time_predict: real_time_predict(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='Entrenar modelo')
    parser.add_argument('-e', '--epochs',            metavar='N',        default=500,              help='Epocas para entrenar', type=int)
    parser.add_argument('-p', '--predict',           action='store_true',                           help='Categorizar archivos de la carpeta predict')
    parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='Categorizar sonido en tiempo real')
    parser.add_argument('-v', '--verbose',           action='store_true',                           help='')
    parser.add_argument('-s', '--log-speed',         action='store_true',                           help='')
    parser.add_argument('-b', '--batch-size',        metavar='size',     default=64,                help='', type=int)
    args = parser.parse_args()
    main(args)
