import os
import pickle

import pandas as pd

from .base_model import BaseReviewModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor


class BoWModel(BaseReviewModel):
    vectorizer: CountVectorizer

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series):
        super().__init__(x_train=x_train, y_train=y_train)

    def init_model(self):
        self.vectorizer = CountVectorizer()
        self.model = RandomForestRegressor(n_jobs=os.cpu_count() - 1, n_estimators=100, max_depth=10)

    def preprocess_x(self, x, fit: bool = False) -> pd.DataFrame:
        if fit:
            # vamos trabalhar somente com os dados textuais
            self.vectorizer.fit(x['review_normalized'])

        return self.vectorizer.transform(x['review_normalized'])

    def preprocess_y(self, y) -> pd.Series:
        return y

    def train(self):
        self.model.fit(X=self.x_train, y=self.y_train)

    def save(self, output_file: os.PathLike):
        with open(output_file, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, input_file: os.PathLike):
        with open(input_file, 'rb') as f:
            self.model = pickle.load(f)
