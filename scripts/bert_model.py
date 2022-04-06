from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
import torch
from tqdm import trange, tqdm
import numpy as np
from enum import Enum
from typing import Optional
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Union
import os

import initializer
from metrics import Metric


class TransformerLanguageDetection:
    def __init__(
            self,
            metric_score_name: Enum = "f1",
            device: Optional[Enum] = None,
            do_save_checkpoint: bool = True,
            load_checkpoint_path: Optional[os.PathLike] = None
    ):
        self.do_save_checkpoint = do_save_checkpoint
        start_time = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.model_save_path = Path(
            initializer.save_checkpoints_path,
            f"{Path(__file__).resolve().stem}_{start_time}.pt"
        )
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.do_save_checkpoint = do_save_checkpoint
        self.model = AutoModelForSequenceClassification.from_pretrained(
            initializer.MODEL_NAME,
            num_labels=initializer.N_LABELS
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(initializer.MODEL_NAME)
        self.get_id_func = lambda x: initializer.languages_set.index(initializer.id_to_lang_dict[x])

        self.metric_score_name = metric_score_name
        self.metric = Metric(metric_score_name)
        self.model = torch.load(load_checkpoint_path, map_location=torch.device('cpu')) if load_checkpoint_path else self.model

    def fit(self, train_loader):
        self.model.train()
        losses = []

        for batch in train_loader:
            text = batch['sentence']
            lang_id = batch['label']

            encoded_text = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            gold = torch.tensor([self.get_id_func(id.item()) for id in lang_id], dtype=torch.long)
            output = self.model(
                encoded_text['input_ids'].to(self.device),
                encoded_text['attention_mask'].to(self.device),
                labels=gold.to(self.device)
            )

            losses.append(output.loss.item())

            output.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss = np.mean(losses).item()
        return train_loss

    def eval(self, valid_loader):
        self.model.eval()
        scores = []
        losses = []

        with torch.no_grad():
            for batch in valid_loader:
                text = batch['sentence']
                lang_id = batch['label']

                encoded_text = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                gold = torch.tensor([self.get_id_func(id.item()) for id in lang_id], dtype=torch.long)
                output = self.model(
                    encoded_text['input_ids'].to(self.device),
                    encoded_text['attention_mask'].to(self.device),
                    labels=gold.to(self.device)
                )
                batch_predictions = torch.argmax(output.logits.detach(), axis=1).cpu()
                barch_f1_score = self.metric(gold.numpy(), batch_predictions)

                losses.append(output.loss.item())
                scores.append(barch_f1_score)

        val_loss = np.mean(losses).item()
        val_score = np.mean(scores).item()
        return val_loss, val_score

    def train(
            self,
            train_dataset,
            epochs: int = 10,
            learning_rate: float = 1e-4,
            train_size: float = 0.9,
            freeze: bool = True,
            batch_size: int = 128
    ):
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        print(batch_size)
        train_len = int(train_size * len(train_dataset))
        train_subset, val_subset = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
        train_loader = DataLoader(train_subset, batch_size=batch_size)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        if freeze:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        best_score = 0
        train_losses = []
        val_losses = []
        val_scores = []
        for epoch in trange(epochs, desc="Training..."):
            train_loss = self.fit(train_loader)
            val_loss, val_score = self.eval(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_scores.append(val_score)

            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'Train loss {train_loss}')
            print(f'Val loss {val_loss} {self.metric_score_name}-score {val_score}')
            print('=' * 50)

            if val_score > best_score:
                self.save_chekpoints()
                best_score = val_score

        if self.do_save_checkpoint:
            self.model = torch.load(self.model_save_path)
        return train_losses, val_losses, val_scores

    def test(
            self,
            test_dataset,
            batch_size: int = 128
    ):
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()
        scores = []

        for batch in tqdm(test_loader, desc="Testing..."):
            text = batch['sentence']
            lang_id = batch['label']

            encoded_text = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            gold = torch.tensor([self.get_id_func(id.item()) for id in lang_id], dtype=torch.long)
            output = self.model(
                encoded_text['input_ids'].to(self.device),
                encoded_text['attention_mask'].to(self.device),
                labels=gold.to(self.device)
            )
            batch_predictions = torch.argmax(output.logits.detach(), axis=1).cpu()
            batch_score = self.metric(gold.numpy(), batch_predictions)
            scores.append(batch_score)

        test_score = np.mean(scores).item()
        print(f'Test {self.metric_score_name}-score {test_score}')

    def predict(self, sentences, threshold: float = 0.1) -> Dict:
        self.model.eval()
        encoded_sent = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        output = self.model(
            encoded_sent['input_ids'].to(self.device),
            encoded_sent['attention_mask'].to(self.device),
            labels=None
        )
        predictions = torch.nn.functional.softmax(output.logits, dim=1)
        if predictions.size()[0] > 1:
            mask = torch.ones(predictions.size()[0])
            mask = 1 - mask.diag()
        else:
            mask = torch.tensor([[1.]])
        sim_vec = torch.nonzero((predictions >= threshold) * mask)
        labels = {(lambda ids: initializer.languages_set[ids[1]])(ids):
                      (sentences[ids[0].item()], predictions[ids[0], ids[1]].item()) for ids in sim_vec}

        return labels

    def save_chekpoints(self) -> None:
        if self.do_save_checkpoint:
            torch.save(self.model, self.model_save_path)
