from bert_model import TransformerLanguageDetection
from logreg_model import NGramsClassifier
import json
import argparse


def main(model_name, path_file=None, write_from_terminal: bool = True):


    # if path_file:
    #     with open(path_file, 'r') as f:
    #         text_data = []
    #         for line in f:
    #             text_data.append(line.split('\n')[0])
    if write_from_terminal:
        text = input("Предложение: ")
    if model_name == 'LOGREG':
        classifier = NGramsClassifier()
        answer = classifier.predict_proba(text)
    elif model_name == 'BERT':
        classifier = TransformerLanguageDetection()
        answer = classifier.predict(text)
    else:
        raise NotImplementedError

    print("For this sentence model have {} predictions: {}".format(len(answer), answer))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--model')
    my_args = parser.parse_args()
    model_type = my_args.model
    # parser.add_argument('-f', '--file')
    # path_to_file = my_args.file
    main(model_name=model_type)
