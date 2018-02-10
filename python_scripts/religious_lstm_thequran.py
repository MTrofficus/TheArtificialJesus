import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import h5py
import json

##########################
###### THE QURAN #########
##########################

"""
directory = "/home/trofficus/Desktop/twitter_project/Conocimiento/quran-verse-by-verse-text"
files = np.sort(os.listdir(directory))

quran_raw = ""
for i in files:
	raw_text = open(directory + "/" + i).read()
	raw_text += "\n"
	quran_raw += raw_text

with open("/home/trofficus/Desktop/twitter_project/Conocimiento/quran.txt", "w") as f:
	f.write(quran_raw)
"""

###############################
#### DATA PREPROCESSING #######
###############################


directory = "/home/trofficus/Desktop/twitter_project/Conocimiento/quran.txt"
quran_raw = open(directory).read()

n_char = len(list(quran_raw)) # 757297

# We have the Quran and The Bible. Now, let's add them up

# Let's study the different words
char_unique = set(list(quran_raw))
nchars = len(list(quran_raw))

# We will build a dictionary for a char and its number id
char_2_num = {x[1]:x[0] for x in enumerate(char_unique)}
num_2_char = {char_2_num[x]:x for x in char_unique}


assert "a" == num_2_char[char_2_num["a"]] # We see if the transformation is correct


# We must decide the secuence we will put into our LSTM
# Let's see some statistics over the size of the new_religion text

versicles = quran_raw.split("\n")
lenght = [len(x) for x in versicles]
average_length=np.mean(lenght) #129 words
median_length=np.median(lenght) # 119 words

# We will take around 150 words to start
# Now we must builda a loop to structure the data

seq_length = 150
dataX = []
dataY = []
for i in range(0, nchars - seq_length, 1):
	seq_in = quran_raw[i:i + seq_length]
	seq_out = quran_raw[i + seq_length]
	dataX.append([char_2_num[char] for char in seq_in])
	dataY.append(char_2_num[seq_out])

# We will insert this 
n_patterns = len(dataX)
n_vocab = len(char_unique)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
Y = np_utils.to_categorical(dataY)

del dataX
del dataY

###############################
######## MODELING ############
##############################

def model_generator(units, dropout1, dropout2, optimizer, activations):
	model = Sequential()
	model.add(LSTM(units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(dropout1))
	model.add(LSTM(units, return_sequences=True))
	model.add(Dropout(dropout2))
	model.add(LSTM(units, return_sequences=True))
	model.add(TimeDistributed(Dense(y.shape[1], activation='softmax')))
	model.compile(loss="categorical_crossentropy", optimizer=optimizer,
				  metrics=["accuracy"])
	return model


###  Let's set a collection of hyperparameters

epochs = [100, 300]
batch_size = [128]
units = [700,1024]
dropout1 = [0.2, 0.3]
dropout2 = [0.2, 0.3]
optimizer = ["Adam"]


param_grid = dict(batch_size=batch_size, epochs=epochs, units=units, 
				  dropout1=dropout1, dropout2=dropout2, optimizer=optimizer)

model = KerasClassifier(build_fn=model_generator, verbose=1)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
grid_result = grid.fit(X, y)

best_score = grid_result.best_score_
best_params = grid_result.best_params_

print("The best model has a score (Categorical crossentropy) of %f using this number of epochs: %s" %(best_score, best_params))

### We write the best selection of hyperparameters to a file

with open("results.txt", "w") as f:
	f.write(str(best_score))
	f.write(str(best_params))

### No we are going to save the model
best_model = grid_result.best_estimator_.model
best_model_file_path = 'best_model'
model2json = best_model.to_json()
with open( best_model_file_path+".json", "w") as json_file:
	json_file.write(model2json)
	best_model.save_weights(best_model_file_path+".h5")



