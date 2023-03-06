# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:53:56 2023

@author: Alkios
"""
import spacy
import sys

sys.path.append("..")
from package import *


# exemple from internet, transforming the data
path = "data/data.csv"

text = """
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.
"""
TEST_REVIEW = """
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)
"""

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
filtered_tokens = [token for token in doc if not token.is_stop]
lemmas = [f"Token: {token}, lemma: {token.lemma_}" for token in filtered_tokens]

if __name__ == "__main__":
    train, test = split_data(path)
    train_model(train, test)
    print("Testing model")
    test_model(TEST_REVIEW)
