import os
import re

import nltk
import pandas as pd
from unidecode import unidecode

nltk.download('stopwords')


def normalize_accents(text: str) -> str:
    """
    Changes accented words to non-accented versions.
    :param text: Text to process.
    :return: Processed text.
    """
    return unidecode(text)


def remove_stopwords(text: str, language: str = 'portuguese') -> str:
    """
    Remoevs stopwords from text using NLTK.
    :param text: text to remove stopwords from.
    :param language: language to process.
    :return: text without stopwords.
    """
    stopwords = nltk.corpus.stopwords.words(language)
    if language == 'portuguese':
        stopwords.remove('não')
    elif language == 'english':
        stopwords.remove('not')
        stopwords.remove('no')

    return ' '.join([w for w in text.split() if w not in stopwords])  # removemos stopwords


def normalize_words(text: str) -> str:
    """
    Normalizes words in a string, keeping only words, numbers, hyphens and spaces.
    :param text: String to process
    :return: Processed string
    """
    return re.sub(r"[^\w\- ]", '', text)  # deixamos apenas palavras, números, hífens e espaços


def normalize_spaces(text: str) -> str:
    """
    Removes extra (double+) spaces from a string.
    :param text: String to process
    :return: Processed string
    """
    return re.sub(r" +", ' ', text).strip()


def normalize_review(review: str) -> str:
    """
    Standard pipeline of preprocessing a single review.
    Makes it lowercase, remove accents, stopwords, invalid words, and normalizes spacing.
    :param review: raw review text
    :return: normalized review
    """
    review = review.lower()
    review = normalize_accents(review)
    review = remove_stopwords(review)
    review = normalize_words(review)
    review = normalize_spaces(review)

    return review


def load_review_data(file_path: os.PathLike, normalize=True) -> (pd.DataFrame, pd.Series):
    """
    Loads review file from specified path, selects columns of interest, and returns X and Y separately.
    :param file_path: csv file containing the reviews
    :param normalize: whether to apply normalization transformations
    :return: (X, Y) tuple with X containing the featues and Y the target column
    """
    df = pd.read_csv(file_path, index_col='index')

    df['review_normalized'] = (df['review_title'] + ' ' + df['review_text'])
    if normalize:
        df['review_normalized'] = df['review_normalized'].map(normalize_review)

    x = df[['review_normalized']]

    y = df['overall_rating']

    return x, y
