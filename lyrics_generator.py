
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
#-------------------------------------------

tokenizer = Tokenizer()

data = open('txt.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

#-----------------------------------------------

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=5, verbose=0, mode='auto')
##history = model.fit(xs, ys, epochs=100, verbose=1)
print(model.summary())
##print(model)

#-------------------------------------------------------------------

checkpoint_path = "lyrics_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
##model.fit(xs, 
##          ys,  
##          epochs=100,
##          callbacks=[cp_callback]) 

#-------------------------------------------------------------------

seed_text = "it's freaking, no it's not, i'm your tensorflow here to sing a song"
next_words = 100
  
for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=max_sequence_len-1,
                                   padding='pre')
        model.load_weights(checkpoint_path) #added for loading checkpoint
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
                if index == predicted:
                        output_word = word
                        break

        seed_text += " " + output_word        
        
	

print(seed_text)
