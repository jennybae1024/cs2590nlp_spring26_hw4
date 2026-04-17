import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

import random
import re

POS_PREFIXES = [
    "I honestly expected to enjoy this. The premise sounded promising and the cast seemed solid. That first impression did not last very long.",
    "At first this looked like something I would probably like. The setup seemed appealing and I was ready to give it a fair chance. That expectation faded pretty quickly.",
    "Going in, I assumed this would be decent. A few details sounded promising at the start. That turned out to be misleading.",
]

NEG_PREFIXES = [
    "At first I thought this would be a mess. The trailer looked dull and I expected very little. That first reaction turned out to be unfair.",
    "I went into this with very low expectations. Honestly, I assumed it would probably disappoint me. I ended up being more wrong than I expected.",
    "Before watching, I had mostly convinced myself that this would not be good. Nothing about it sounded especially appealing at first. That assumption did not hold.",
]

POS_MIDDLES = [
    "For a brief moment, I thought it might actually work.",
    "Some early parts made it seem more promising than it really was.",
    "There were isolated moments that almost made me think it would be decent.",
]

NEG_MIDDLES = [
    "The opening made me think this might go badly.",
    "Early on, I honestly thought I would not enjoy it.",
    "At first, I was fairly skeptical about where this was going.",
]

HEDGES = [
    "personally",
    "to be honest",
    "for what it is worth",
    "in my opinion",
    "at least for me",
    "I mean",
    "to be fair",
]

ASIDES = [
    "(or so I thought at first)",
    "(at least at the beginning)",
    "(which was my first reaction anyway)",
    "(or that was how it initially seemed)",
]

def _split_into_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def reorder_sentences_light(text):
    sentences = _split_into_sentences(text)

    if len(sentences) < 4:
        return text

    max_idx = min(4, len(sentences) - 2)
    i = random.randint(1, max_idx)

    sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
    return " ".join(sentences)


def custom_transform(example):
    if not isinstance(example.get("text"), str):
        return example

    text = example["text"].strip()
    if not text:
        return example

    label = example.get("label", None)

    if label == 1:
        transformed = random.choice(NEG_PREFIXES) + " " + text
        middle_pool = NEG_MIDDLES
    elif label == 0:
        transformed = random.choice(POS_PREFIXES) + " " + text
        middle_pool = POS_MIDDLES
    else:
        transformed = "I had mixed expectations before watching this. " + text
        middle_pool = ["I was not completely sure what to expect at first."]

    sentences = _split_into_sentences(transformed)

    if len(sentences) >= 2:
        insert_sent = random.choice(middle_pool)
        insert_idx = 1 if len(sentences) < 4 else 2
        sentences.insert(insert_idx, insert_sent)

    if len(sentences) >= 4:
        insert_sent = random.choice(middle_pool)
        insert_idx = min(3, len(sentences) - 1)
        sentences.insert(insert_idx, insert_sent)

    transformed = " ".join(sentences)

    hedge = random.choice(HEDGES)
    positions = [m.end() for m in re.finditer(r"[,.!?]", transformed)]
    if positions:
        pos = positions[min(2, len(positions) - 1)]
        transformed = transformed[:pos] + f" {hedge}," + transformed[pos:]

    aside = random.choice(ASIDES)
    positions = [m.end() for m in re.finditer(r"[,.!?]", transformed)]
    if positions:
        pos = positions[min(2, len(positions) - 1)]
        transformed = transformed[:pos] + " " + aside + transformed[pos:]

    # if random.random() < 0.8:
    #     transformed = reorder_sentences_light(transformed)
    transformed = reorder_sentences_light(transformed)
    transformed = reorder_sentences_light(transformed)


    def introduce_typo(word):
        if len(word) <= 4:
            return word
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]

    if random.random() < 0.5:
        words = transformed.split()
        idx = random.randint(0, len(words) - 1)

        clean = re.sub(r'[^\w]', '', words[idx])

        if len(clean) > 4:
            typo_word = introduce_typo(clean)
            words[idx] = words[idx].replace(clean, typo_word)

        transformed = " ".join(words)

    transformed = transformed.replace(". ", " -- ", 1)

    transformed = re.sub(r"\s+", " ", transformed).strip()
    example["text"] = transformed
    return example
