from ngram_model import NGramsClassifier
import time
import configparser
from bert_model import TransformerLanguageDetection
from dataset import WILI2018Dataset


def main():

    start_time = time.time()
    config = configparser.ConfigParser()
    config.read('scripts/config.ini')
    train_set = WILI2018Dataset("train")
    test_set = WILI2018Dataset("test")
    if config['PIPELINE']['mode'] == 'NGRAM':
        classifier = NGramsClassifier()
        classifier.train(list(train_set))
        f1_score = classifier.test(list(test_set))
        print("f1 score on test dataset", f1_score)
    elif config['PIPELINE']['mode'] == 'BERT':
        classifier = TransformerLanguageDetection(device=config['BERT']['device'])
        classifier.train(train_set, batch_size=int(config['BERT']['batch_size']),
                         epochs=int(config['BERT']['epochs']))
        classifier.test(test_set, batch_size=int(config['BERT']['batch_size']))
        classifier.save_chekpoints()
    else:
        raise NotImplementedError

    classifier.save_chekpoints()
    print("checkpoint are saved to results folder")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
