import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# Mediante los wrappers aplicamos GridSeach de SL a Keras
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV


########################################
###### DATA PREPROCESSING ##############
########################################

# load ascii text and covert to lowercase
filename = "/home/trofficus/Desktop/twitter_project/Conocimiento/alice.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


#############################################
##### MODELING WITH LSTM (2 Layers) #########
#############################################


# We create a function to apply GridSearch

def model_generator(units, dropout, optimizer, activations):
	model = Sequential()
	model.add(LSTM(units, input_shape=(X.shape[1], X.shape[2]),
						 activation=activations, return_sequences=True))
	model.add(Dropout(dropout))
	model.add(LSTM(units, activation = activations, return_sequences=True))
	model.add(Dropout(dropout))
	model.add(Dense(y.shape[1], activation='softmax'))
	
	model.compile(loss="categorical_crossentropy", optimizer=optimizer,
				  metrics=["accuracy"])
	return model


# Let's set a collection of hyperparameters

epochs = [20, 50, 100]
batch_size = [30, 60, 100]
units = [128, 256, 512]
activations = ["relu", "tanh", "elu"]
dropout = [0.2, 0.3, 0.4]
optimizer = ["RMSprop", "Adam"]


param_grid = dict(batch_size=batch_size, epochs=epochs, units=units, 
				  activations=activations, dropout=dropout,
				  optimizer=optimizer)

model = KerasClassifier(build_fn=model_generator, verbose=1)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
grid_result = grid.fit(X, y)

best_score = grid_result.best_score_
best_params = grid_result.best_params_

print("The best model has a score (Categorical crossentropy) of %f using this number of epochs: %s" %(best_score, best_params))

# Escribamos los resultados a un fucking fichero

with open("results.txt", "w") as f:
	f.write(str(best_score))
	f.write(str(best_params))
