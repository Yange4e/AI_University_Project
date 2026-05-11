import json
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # ДОБАВЬ ЭТУ СТРОКУ
nltk.download('punkt')
stemmer = SnowballStemmer("russian")

# Загрузка
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words, classes, documents = [], [], []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set([stemmer.stem(w.lower()) for w in words if w not in ignore_letters])))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Подготовка данных
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

np.random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Нейросеть
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("\n[OK] Модель обучена. Файлы chatbot_model.h5, words.pkl, classes.pkl созданы.")