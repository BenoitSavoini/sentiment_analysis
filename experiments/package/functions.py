# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:53:56 2023

@author: Alkios
"""

import spacy
import random
import pandas
from spacy.util import minibatch, compounding

def load_training_data(data_directory):
    """Loading of the dataset from a directory.

    Args:
        data_directory (str): directory of the dataset

    Returns:
        tweets (list): List of tweets
        label (list): List of labels
    """
    tweets = []
    label = []
    colnames = ["target", "ids", "date", "flag", "user", "text"]
    df = pandas.read_csv(data_directory, names=colnames, encoding="latin-1")
    for i in range(len(df)):
        text = df["text"][i]
        nltk_label = df["target"][i]
        tweets.append(text)
        label.append(nltk_label)

    return tweets, label

def split_data(data_directory: str, limit=0, split=0.8):
    """Train-test split of the dataset from a directory.

    Args:
        data_directory (str): directory of the dataset
        limit (int, optional): _description_. Defaults to 0.
        split (float, optional): Spliting ratio between training and test data. Defaults to 0.8.

    Returns:
        train (list): train data
        test (list): test data
    """
    reviews = []
    colnames = ["target", "ids", "date", "flag", "user", "text"]
    df = pandas.read_csv(data_directory, names=colnames, encoding="latin-1")
    for i in range(len(df)):
        text = df["text"][i]
        spacy_label = df["target"][i]
        reviews.append((text, spacy_label))
    random.shuffle(reviews)
    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    train = reviews[:split]
    test = reviews[split:]
    return train, test

def train_model(training_data, test_data, iterations=20):
    """Model training

    Args:
        training_data (list): Training data
        test_data (list): Test data
        iterations (int, optional): Number of iterations. Defaults to 20.
    """
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer, textcat=textcat, test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def evaluate_model(tokenizer, textcat, test_data):
    """Evaluation of the model.

    Args:
        tokenizer (_type_): directory of the dataset
        textcat (_type_): _description_. Defaults to 0.
        test_data (list): Test data. Defaults to 0.8.

    Returns:
        Dictionary giving the precision, the recall, and the F1-score of the model
    """
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if predicted_label == "neg":
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def test_model(input_data: str):
    """Test of the model.

    Args:
        input_data (str, optional): Sentence to test.
    """
    # Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

