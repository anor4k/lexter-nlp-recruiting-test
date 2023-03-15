import os

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm.auto import tqdm
from src.models.base_model import BaseReviewModel


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Ber2WDataset(Dataset):
    def __init__(self, x, y, model_size='base'):
        self.tokenizer = AutoTokenizer.from_pretrained(f'neuralmind/bert-{model_size}-portuguese-cased')

        reviews = x['review_normalized'].to_list()

        tokens = self.tokenize_review(reviews)

        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.ratings = torch.Tensor(y.to_list())

    def tokenize_review(self, reviews):
        # max_length = 120 porque 99% do nosso dataset tem até 120 palavras.
        # token != palavra mas é uma referência decente
        return self.tokenizer(reviews, return_tensors='pt', padding=True, truncation=True, max_length=120)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        rating = self.ratings[idx]

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'rating': rating
        }

        return sample


class Ber2WModel(torch.nn.Module):
    def __init__(self, model_size: str = 'base'):
        assert model_size in ['base', 'large']
        super().__init__()

        model_path = f'neuralmind/bert-{model_size}-portuguese-cased'

        self.bert = AutoModel.from_pretrained(model_path)

        for param in self.bert.parameters():
            # vamos treinar somente a nova camada linear
            param.requires_grad = False

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear1 = torch.nn.Linear(self.bert.config.hidden_size, 512)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0]

        x = self.dropout(hidden)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        output = self.output_layer(x)

        return output.squeeze()

    def predict(self, tokens):
        self.eval()
        return self.forward(tokens['input_ids'], tokens['attention_mask']).cpu().detach().numpy()


class Ber2W(BaseReviewModel):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler

    def __init__(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame,
            y_test: pd.DataFrame,
            model_size: str = 'base',
            batch_size: int = 64
    ):
        self.model_size = model_size
        self.device = get_device()
        super().__init__(x_train=x_train, y_train=y_train)

        self.train_dataset = Ber2WDataset(x=self.x_train, y=self.y_train, model_size=self.model_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size)

        self.val_dataset = Ber2WDataset(x=x_test, y=y_test, model_size=self.model_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

    def init_model(self):
        self.model = Ber2WModel(model_size=self.model_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

    def preprocess_x(self, x, fit: bool = False) -> pd.DataFrame:
        if fit:
            return x
        else:
            return self.train_dataset.tokenize_review(x['review_normalized'].tolist()).to(self.device)

    def preprocess_y(self, y) -> pd.Series:
        return y

    def train_step(self):
        self.model.train(True)

        running_loss = 0
        correct_preds = 0

        for i, batch in enumerate(tqdm(self.train_loader, unit='batches'), 0):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            rating = batch['rating'].squeeze().to(self.device)

            self.optimizer.zero_grad()

            pred = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
            correct = (pred.cpu().detach().numpy().round() == rating.cpu().detach().numpy()).sum()

            loss = mse_loss(pred, rating)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save loss data
            running_loss += loss.item()
            correct_preds += correct
            if i % 100 == 0 and i > 0:
                print(f'Batch: {i:>5,}:', end=' ')
                print(f'Loss (train): {running_loss / i:>5.2f}.', end=' ')
                print(f'Accuracy (train): {correct_preds / (i * self.train_loader.batch_size):5.2%}')

        print(f"Training Running Loss: {running_loss / len(self.train_loader):.2f}")
        print(f"Training Running Accuracy: {correct_preds / len(self.train_dataset):.2%}")

        return running_loss

    def validation_step(self):
        self.model.eval()

        running_loss = 0
        correct_preds = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, unit='batches'), 1):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                rating = batch['rating'].squeeze().to(self.device)

                pred = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)

                loss = mse_loss(pred, rating)
                correct = (pred.cpu().detach().numpy().round() == rating.cpu().detach().numpy()).sum()

                # save loss data
                running_loss += loss.item()
                correct_preds += correct

        print(f"Validation Running Loss: {running_loss / len(self.val_loader):.2f}")
        print(f"Validation Running Accuracy: {correct_preds / len(self.val_dataset):.2%}")

    def train(self, epochs: int = 5):
        for epoch in range(epochs):
            print(f"Training: epoch {epoch}")
            self.train_step()
            self.validation_step()

    def save(self, output_file: os.PathLike):
        torch.save(self.model.state_dict(), output_file)

    def load(self, input_file: os.PathLike):
        self.model.load_state_dict(torch.load(input_file))
