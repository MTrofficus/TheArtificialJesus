# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:00:26 2018

@author: EXT94908
"""

### Tutorial 1: Alicia en el país de las maravillas 


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import h5py
from sklearn.preprocessing import LabelEncoder

filename = "alice.txt.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# Ahora vamos a seleccionar los caracteres distintos
chars = sorted(list(set(raw_text)))

# Como primera aproximación, vamos a transformar los caracteres a enteros

# Codificamos los 60 caracteres a enteros
le = LabelEncoder()
le.fit(chars)

# le.inverse_transform nos permite hacer la transformacion inversa

# Ahora vamos a ver qué secuencia introducimos en la LSTM

seq_length = 100 # Aquí lo que haremos será determinar un tamaño de secuencia (lo que introducimos en la red)
n_chars = len(raw_text)
n_chars_unique = len(chars)

dataX = []
dataY = []

for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i+seq_length] # Es la secuencia de entrada (tamaño seq_length como es obvio)
    seq_out = raw_text[i+seq_length] # En este caso sólo sacaremos un valor (una letra)
    dataX.append(le.transform(list(seq_in)))
    dataY.append(le.transform(list(seq_out)))


X = np.array(dataX)
Y = np.array(dataY)
# Así pues, tenemos un array de 100 valores como entrada, y uno de salida,
# Para poder aplicar la categorizacion, necesitamos aplicar one hot encoding a los valores de salida.
# Además, estructuraremos los arrays de tal modo que quepan bien en el modelo

n_patterns = X.shape[0]
X = np.reshape(X, (n_patterns, seq_length, 1))
# Además, debemos normalizarlos
X = X/float(n_chars_unique)

# Hacemos one hot encode a la variable de salida
Y = np_utils.to_categorical(Y)

#### Definimos el modelo LSTM

n_cells = 256

model = Sequential()
model.add(LSTM(n_cells, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer="adam")

# Además, vamos a ir guardaando los pesos del modelo cada vez que se produzca una disminucion de la funcion de coste
"""
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
"""
#model.fit(X,Y, epochs = 1, batch_size = 128, callbacks = callbacks_list)

model.fit(X,Y, epochs = 1, batch_size = 128)
















