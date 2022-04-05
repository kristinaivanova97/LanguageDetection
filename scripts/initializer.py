import pandas as pd
from pathlib import Path
from datasets import load_dataset
import os


home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")
save_checkpoints_path = Path(home_path, "results/")
os.makedirs(data_folder, exist_ok = True)
os.makedirs(save_checkpoints_path, exist_ok = True)

if not Path(Path(data_folder, "labels.csv")).is_file():
    import wget
    import shutil
    wget.download("https://zenodo.org/record/841984/files/wili-2018.zip?download=1", "dataset.zip")
    shutil.unpack_archive("dataset.zip", data_folder)
wili2018 = load_dataset("wili_2018")
wili2018_langs_labels = wili2018['train'].features['label'].names

languages_set = ["rus", "ukr", "bel", "kaz", "aze", "hye", "kat", "heb", "eng", "deu"]
N_LABELS = len(languages_set)

annotation_df = pd.read_csv(Path(data_folder, "labels.csv"), sep=";")
annotation_df['id'] = annotation_df['Label'].apply(lambda x: wili2018_langs_labels.index(x) if x in wili2018_langs_labels else -1)
lang_to_id_dict = dict(zip(annotation_df.Label, annotation_df.id))
id_to_lang_dict = dict(zip(annotation_df.id, annotation_df.Label))

MODEL_NAME = "bert-base-multilingual-cased"
