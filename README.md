# SentimentEmojiPredict
Sentiment analysis using RNN,LSTM,BILSTM,GRU
#Import laiberies
import numpy as np
import pandas as pd
import emoji
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import SimpleRNN
from keras.layers import LSTM, Bidirectional

# Load the data
mapping = pd.read_csv("D:\Mapping.csv")
output = pd.read_csv("D:\OutputFormat.csv")
train = pd.read_csv("D:\Train.csv")
test = pd.read_csv("D:\Test.csv")


train.head()

test.tail()

mapping.head()

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['TEXT'])
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(train['TEXT'])
test_sequences = tokenizer.texts_to_sequences(test['TEXT'])
max_seq_length = 100  # define the maximum sequence length
train_data = pad_sequences(train_sequences, maxlen=max_seq_length)
test_data = pad_sequences(test_sequences, maxlen=max_seq_length)


train_data

test_data

# Check unique values in the label column
unique_labels = train['Label'].unique()
num_classes = len(unique_labels)

# Preprocess labels
label_mapping = {label: index for index, label in enumerate(unique_labels)}
train_labels = train['Label'].map(label_mapping)
train_labels = to_categorical(train_labels, num_classes=num_classes)

# Model BILSTM 
model1 = Sequential()
model1.add(Embedding(vocab_size, 100, input_length=max_seq_length))
model1.add(Bidirectional(LSTM(128)))
model1.add(Dense(num_classes, activation='softmax'))
# Compile and train the model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(train_data, train_labels, epochs=3, batch_size=32)


#model RNN
model2 = Sequential()
model2.add(Embedding(vocab_size, 100, input_length=max_seq_length))
model2.add(SimpleRNN(128))
model2.add(Dense(num_classes, activation='softmax'))


# Compile and train the model
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(train_data, train_labels, epochs=3, batch_size=32)

# Define the GRU model
model3 = Sequential()
model3.add(Embedding(vocab_size, 100, input_length=max_seq_length))
model3.add(GRU(128))
model3.add(Dense(num_classes, activation='softmax'))

# Compile and train the model
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(train_data, train_labels, epochs=3, batch_size=32)



# Define the LSTM model
model4 = Sequential()
model4.add(Embedding(vocab_size, 100, input_length=max_seq_length))
model4.add(LSTM(128))
model4.add(Dense(num_classes, activation='softmax'))
# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=3, batch_size=32)

#MODEL1
predictions1 = np.argmax(model1.predict(test_data), axis=1)

#MODEL2
predictions2 = np.argmax(model2.predict(test_data), axis=1)

#MODEL3
predictions3 = np.argmax(model3.predict(test_data), axis=1)

#MODEL4
predictions4 = np.argmax(model4.predict(test_data), axis=1)

#MODEL1
predicted_emojis1 = mapping['emoticons'][predictions1]

# Loop through each sentence in the test data
for sentence, predicted_emoji in zip(test['TEXT'], predicted_emojis1):
    # Print the sentence and its corresponding emoji
    print(f"Sentence: {sentence}")
    print(f"Predicted Emoji: {predicted_emoji}")
    print()


#MODEL2
predicted_emojis2 = mapping['emoticons'][predictions2]


# Loop through each sentence in the test data
for sentence, predicted_emoji in zip(test['TEXT'], predicted_emojis2):
    # Print the sentence and its corresponding emoji
    print(f"Sentence: {sentence}")
    print(f"Predicted Emoji: {predicted_emoji}")
    print()


#MODEL3
predicted_emojis3 = mapping['emoticons'][predictions3]



# Loop through each sentence in the test data
for sentence, predicted_emoji in zip(test['TEXT'], predicted_emojis3):
    # Print the sentence and its corresponding emoji
    print(f"Sentence: {sentence}")
    print(f"Predicted Emoji: {predicted_emoji}")
    print()


#MODEL4
predicted_emojis4 = mapping['emoticons'][predictions4]


# Loop through each sentence in the test data
for sentence, predicted_emoji in zip(test['TEXT'], predicted_emojis4):
    # Print the sentence and its corresponding emoji
    print(f"Sentence: {sentence}")
    print(f"Predicted Emoji: {predicted_emoji}")
    print()

