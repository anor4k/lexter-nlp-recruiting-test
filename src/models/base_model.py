import os
from abc import abstractmethod
from typing import Any

import pandas as pd
from sklearn.metrics import mean_squared_error


class BaseReviewModel:
    model: Any

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.init_model()
        self.x_train = self.preprocess_x(x_train, fit=True)
        self.y_train = self.preprocess_y(y_train)

    @abstractmethod
    def preprocess_x(self, x, fit: bool = False) -> pd.DataFrame:
        raise NotImplementedError("Define preprocessing of features here")

    @abstractmethod
    def preprocess_y(self, y) -> pd.Series:
        raise NotImplementedError("Define preprocessing of target here")

    @abstractmethod
    def init_model(self):
        raise NotImplementedError("Please define model initialization.")

    @abstractmethod
    def train(self):
        raise NotImplementedError("Please implement a train method for the model.")

    @abstractmethod
    def save(self, output_file: os.PathLike):
        raise NotImplementedError("Please implement a save method for the model.")

    @abstractmethod
    def load(self, input_file: os.PathLike):
        raise NotImplementedError("Please implement a load method for the model.")

    def predict(self, x_pred: pd.DataFrame | pd.Series) -> pd.Series:
        x_pred_prep = self.preprocess_x(x_pred, fit=False)
        y_pred = self.model.predict(x_pred_prep).round()  # nosso target é sempre inteiro
        y_pred = pd.Series(y_pred, index=x_pred.index)
        y_pred = y_pred.map(lambda x: max(min(x, 5), 1))  # garante o target entre 1 e 5

        return y_pred

    def predict_and_report(self, x_pred: pd.DataFrame, y_true: pd.Series) -> pd.Series:
        y_pred = self.predict(x_pred)
        y_true = self.preprocess_y(y_true)
        mse = mean_squared_error(y_true, y_pred)
        accuracy = (y_true == y_pred).astype('int').sum() / len(y_true)

        report = f"Accuracy: {accuracy:.2%}"
        report += f"\n    MSE: {mse:< 6.4f}"
        print(report)

        return y_pred
