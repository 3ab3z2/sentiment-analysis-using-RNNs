import numpy as np
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Ensure 'text' column is string and handle NaN values
    data['text'] = data['text'].fillna('').astype(str)
    # Replace any extra characters with a dot
    data['text'] = re.sub(r'[,!?;-]+', '.', data['text'])
    # Repalce all captials to lowers
    data['text'] = [ ch.lower() for ch in data['text']
         if ch.isalpha()
         or ch == '.'
       ]
    texts = data['text'].values
    labels = data['label'].values
    return texts, labels

# Paths to datasets
train_dataset_path = 'archive/generic_sentiment_dataset_50k.csv'
test_dataset_path = 'archive/generic_sentiment_dataset_10k.csv'

# Load datasets
train_texts, train_labels = load_data(train_dataset_path)
test_texts, test_labels = load_data(test_dataset_path)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    SimpleRNN(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=5, validation_data=(test_padded, test_labels), batch_size=32)

# Save the model
model.save('sentiment_rnn_model.h5')

# Load the model for evaluation
def evaluate_input():
    model = tf.keras.models.load_model('sentiment_rnn_model.h5')
    while True:
        user_input = input("Enter a sentence for sentiment analysis (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=100, padding='post', truncating='post')
        prediction = model.predict(input_padded)
        sentiment = np.argmax(prediction)
        sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Sentiment: {sentiment_label[sentiment]}")

# Run evaluation
if __name__ == "__main__":
    evaluate_input()