# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:17:43 2023

@author: Alkios
"""
import numpy as np
from os import path
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import (
    Embedding,
    Dense,
    LSTM,
)
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tensorflow as tf
import sys 
sys.path.append('..')
from package.functions import *

print(tf.__version__)
tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

data_path = "data/data.csv"
model_path = 'model/my_h5_model_pad50.h5'
max_words = 10000

# Textes de tweets d'entraînement et labels correspondants
def run_experiment():
    """Running of the sentiment analysis project
    """
    tweets, labels = load_training_data(data_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        tweets, labels, test_size=0.2, random_state=42
    )

    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)

    new_train_labels = np.where(train_labels == 0, 0, np.where(train_labels == 2, 1, 2))
    new_test_labels = np.where(test_labels == 0, 0, np.where(test_labels == 2, 1, 2))

    # Paramètres du tokenizer
    max_words = 10000  # Nombre maximal de mots à prendre en compte dans le tokenizer
    max_len = 50  # Longueur maximale des séquences de mots

    # Création du tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)

    # Conversion des textes de tweets en séquences de mots
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Remplissage des séquences de mots pour qu'elles aient toutes la même longueur
    train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # Création du modèle
    model = Sequential()
    model.add(Embedding(max_words, 128))
    model.add(LSTM(128, dropout=0, recurrent_dropout=0, activation="tanh"))
    model.add(Dense(3, activation="softmax"))

    # Compilation du modèle
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Entraînement du modèle si pas encore fait
    if path.exists(model_path) is False:
        model.fit(train_sequences, new_train_labels, epochs=50, batch_size=128)
        model.save(model_path)
    else:
        model = load_model(model_path)

    loss, acc = model.evaluate(test_sequences, new_test_labels, verbose=2)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
    
    # Exemple d'application
    tweet = "hate it !!! you're shit !"
    # Convertion du tweet en vecteur
    vector = tokenizer.texts_to_sequences([tweet])
    # Remplissage du vecteur avec des zéros pour qu'il ait la même longueur que les autres tweets
    vector = pad_sequences(vector, padding="post", maxlen=max_len)
    # Effectuer la prédiction
    prediction = model.predict(vector)
    print(prediction)

if __name__ == "__main__":
    run_experiment()