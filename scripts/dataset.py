from torch.utils.data import Dataset
from enum import Enum
from typing import List, Tuple

import initializer


class WILI2018Dataset(Dataset):
    def __init__(self, stage: Enum):
        self.original_dataset = initializer.wili2018
        self.stage = stage
        self.texts, self.labels = self.__form_data()

    def __form_data(self) -> Tuple[List[str], List[int]]:
        stage_data = ([], [])
        for sample in self.original_dataset[self.stage]:
            text = sample['sentence']
            label = sample['label']
            if label in initializer.id_to_lang_dict and \
                initializer.id_to_lang_dict[label] in initializer.languages_set:
                stage_data[0].append(text)
                stage_data[1].append(label)
        return stage_data

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        target = self.labels[idx]
        return {
          'sentence': text,
          'label': target
        }