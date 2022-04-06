from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Dict, Union
from enum import Enum
import os
from pathlib import Path
from joblib import dump, load
from datetime import datetime

import initializer
from metrics import Metric


class NGramsClassifier:
    def __init__(
        self,
        metric_score_name: Enum = "f1",
        do_save_checkpoint: bool = True,
        load_checkpoint_path: Optional[os.PathLike] = None
    ):
        self.do_save_checkpoint = do_save_checkpoint
        start_time = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.metric_score_name = metric_score_name
        self.metric = Metric(metric_score_name)

        self.model = load(load_checkpoint_path) if load_checkpoint_path else None
        self.model_save_path = Path(
            initializer.save_checkpoints_path,
            f"{Path(__file__).resolve().stem}_{start_time}.joblib"
        )

    def train(self, train_dataset: List[Dict[str, Union[str, int]]]) -> None:
        x_train, y_train = self.get_right_format_dataset(train_dataset)
        vectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer='char')

        self.model = pipeline.Pipeline([
            ('vectorizer', vectorizer),
            ('clf', LogisticRegression())   
        ])
        self.model.fit(x_train, y_train)
        self.save_chekpoints()

    def get_right_format_dataset(self, data: List[Dict[str, Union[str, int]]]):
        x, y = [], []
        for sample in data:
            x.append(sample['sentence'])
            y.append(sample['label'])
        return x, y

    def predict(self, texts: List[str]) -> List[int]:
        return self.model.predict(texts)

    def predict_proba(self, texts: List[str], threshold: float = 0.2) -> List[dict]:
        predictions = self.model.predict_proba(texts)
        result = [{initializer.id_to_lang_dict[self.model.classes_[i]]: prob for i, prob in enumerate(prediction) if prob > threshold}
                  for prediction in predictions]
        return result

    def test(self, test_dataset: List[Dict[str, Union[str, int]]]) -> float:
        x_test, y_test = self.get_right_format_dataset(test_dataset)
        predicted_labels = self.predict(x_test)
        return self.metric(y_test, predicted_labels)

    def save_chekpoints(self) -> None:
        if self.do_save_checkpoint:
            dump(self.model, self.model_save_path) 
