import numpy as np 
import os
from keras.utils import np_utils


### The Bible
directory = "/home/trofficus/Desktop/twitter_project/Conocimiento/bible.txt"
bible_raw = open(directory).read()

n_char = len(list(bible_raw)) # 4047392

### The Quran

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

bible_raw = bible_raw[:-1]

# Let's study the different words
char_unique = set(list(bible_raw))
nchars = len(list(bible_raw))

# We will build a dictionary for a char and its number id
char_2_num = {x[1]:x[0] for x in enumerate(char_unique)}
num_2_char = {char_2_num[x]:x for x in char_unique}


assert "a" == num_2_char[char_2_num["a"]] # We see if the transformation is correct


# We must decide the secuence we will put into our LSTM
# Let's see some statistics over the size of the new_religion text

versicles = bible_raw.split("\n")
lenght = [len(x) for x in versicles]
average_length=np.mean(lenght) #132 words
median_length=np.median(lenght) # 123 words

# Usually we use 200-300 seq_length. We will put 300
# Now we must builda a loop to structure the data

seq_length = 300
dataX = []
dataY = []
for i in range(0, nchars - seq_length, 1):
	seq_in = bible_raw[i:i + seq_length]
	seq_out = bible_raw[i + seq_length]
	dataX.append([char_2_num[char] for char in seq_in])
	dataY.append(char_2_num[seq_out])

# We will insert this 
n_patterns = len(dataX)
n_vocab = len(char_unique)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
Y = np_utils.to_categorical(dataY)

np.save("/", X)
np.save("/", Y)







