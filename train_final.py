import os, gc, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(r"C:\Users\nagal\OneDrive\Desktop\mentalhealth")

df = pd.read_csv("mental_health.csv")
df = df.dropna(subset=["posts"])
df["posts"] = df["posts"].astype(str)

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

stop_words = set(stopwords.words("english"))
keep_words = {
    "not", "no", "nor", "never",
    "don't", "didn't", "doesn't",
    "can't", "won't", "isn't",
    "aren't", "wasn't", "weren't",
    "couldn't", "shouldn't", "wouldn't",
    "haven't", "hasn't", "hadn't"
}
custom_stopwords = stop_words - keep_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.lower() not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(w, pos="v") for w in tokens]
    return " ".join(tokens)

df["posts"] = df["posts"].apply(preprocess_text)
X = df["posts"]
y = df["intensity"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 100
NUM_WORDS = 10000

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

print(f"Class distribution - train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")

print("Loading GloVe...")
embeddings_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = vector

embedding_dim = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    emb = embeddings_index.get(word)
    if emb is not None:
        embedding_matrix[i] = emb
del embeddings_index
gc.collect()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], trainable=False))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train)), activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training...")
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test), class_weight="balanced")

print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

models_dir = Path(r"C:\Users\nagal\OneDrive\Desktop\mentalhealth\Mental_Health_Flask\models")
models_dir.mkdir(parents=True, exist_ok=True)
model.save(str(models_dir / "glove_gru.keras"))
with open(str(models_dir / "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
with open(str(models_dir / "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
print("Saved model artifacts.")