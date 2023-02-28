# Sentiment Analysis Project - Lucas Ho & Benoit Savoini

The following repository is a first project on sentiment analysis problem by using sentiment140 dataset from Kaggle, containing 1,600,000 tweets extracted using the twitter api (https://www.kaggle.com/datasets/kazanova/sentiment140)

## Content

We focus on the resolution of a sentiment analysis problem in two ways: One by using a LSTM model *lstm_application.py*. And another one by using spacy pipeline *spacy_application.py*.

## Requirements

- keras == 2.11.0
- tensorflow == 2.11.0
- numpy == 1.23.1
- spacy == 3.5.0
- scikit-learn == 1.1.2

Some specific requirements are needed. First, the file *data.csv* has to be unzip from the *data.zip* in the *data* folder. Secondly, you have to download the trained pipeline *en_core_web_sm* by writing in the terminal:

```sh
  python -m spacy download en_core_web_sm
  ```
