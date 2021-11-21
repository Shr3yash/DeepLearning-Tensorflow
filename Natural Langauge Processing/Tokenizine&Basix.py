# in this code tokenizing a string, padding it and printing test sequence are attained.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'this is a demo sentence',
    'another one',
    'this is train sentence',
    'sunio doreamoin ali'
]

tokenizer_variable = Tokenizer(num_words=100, oov_token="<bye>")
tokenizer_variable.fit_on_texts(sentences)
word_index_var = tokenizer_variable.word_index
print(word_index_var)

sequencesss= tokenizer_variable.texts_to_sequences(sentences)

padded = pad_sequences(sequencesss, padding='pre', maxlen=16, truncating='pre')
# we use truncating for deletion of sentences, while we put the sentence through maxlen parameter,
# padding adds zeroes and maxlen is self explanatory uk

print(sequencesss)
print(padded)

test_data = [
    'this is test data',
    'another test sentence',
    'out of bounds demo'
]

test_sequence = tokenizer_variable.texts_to_sequences(test_data)
print(test_sequence)

padded_test = pad_sequences(test_data, padding='pre', maxlen=100)  #truncating='post')
print(padded_test)
